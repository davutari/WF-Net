"""
Model factory: 10 baselines (via timm) + the proposed WF-Net and its ablation variants.

All models are ImageNet-pretrained and expose the same (logits) interface so
they can be trained under an identical protocol.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

import config


# --------------------------------------------------------------------------- #
# CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2,
                              bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# --------------------------------------------------------------------------- #
# SE: Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)
# Channel-only, no spatial 7x7 conv -> fewer params than CBAM.
# --------------------------------------------------------------------------- #
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = torch.sigmoid(self.fc(s)).view(b, c, 1, 1)
        return x * s


def make_attention(name: str, channels: int) -> nn.Module:
    """In-branch feature-refinement attention: 'cbam' | 'se' | 'none'."""
    name = (name or "none").lower()
    if name == "cbam":
        return CBAM(channels)
    if name == "se":
        return SEBlock(channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown attention type: {name!r}")


# --------------------------------------------------------------------------- #
# Proposed model: WF-Net (Waste Fusion Network)
# --------------------------------------------------------------------------- #
def _to_bchw(feat: torch.Tensor, channels: int) -> torch.Tensor:
    """Normalize any backbone feature output to [B, C, H, W]."""
    if feat.dim() == 4:
        if feat.shape[1] == channels:          # already [B, C, H, W]
            return feat
        if feat.shape[-1] == channels:         # [B, H, W, C] (e.g. Swin)
            return feat.permute(0, 3, 1, 2).contiguous()
    if feat.dim() == 3:                        # [B, N, C] tokens -> square map
        b, n, c = feat.shape
        s = int(round(n ** 0.5))
        return feat.transpose(1, 2).reshape(b, c, s, s).contiguous()
    raise ValueError(f"Unexpected feature shape {tuple(feat.shape)}")


class BranchEncoder(nn.Module):
    """One backbone branch: features -> attention refinement -> projected tokens."""

    def __init__(self, backbone_name, out_dim, pretrained=True, attn="cbam"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.channels = self.backbone.num_features
        self.attn = make_attention(attn, self.channels)
        self.proj = nn.Linear(self.channels, out_dim)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        if f.dim() == 3:          # ViT-style token output: drop CLS/distillation tokens
            f = f[:, getattr(self.backbone, "num_prefix_tokens", 0):]
        f = _to_bchw(f, self.channels)
        f = self.attn(f)                                   # [B, C, H, W]
        self._last_map = f                                 # cached for Grad-CAM
        b, c, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2)              # [B, HW, C]
        return self.proj(tokens)                           # [B, HW, out_dim]


class FusionBlock(nn.Module):
    """Pre-norm Transformer encoder block with stochastic depth (DropPath).

    Equivalent to ``nn.TransformerEncoderLayer(norm_first=True)`` but adds a
    per-sample DropPath on each residual branch -- a strong regularizer for a
    from-scratch fusion module trained on a small dataset.
    """

    def __init__(self, dim, heads, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        a, attw = self.attn(h, h, h, need_weights=return_attn,
                            average_attn_weights=True)
        x = x + self.drop_path(a)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attw) if return_attn else x


class CrossAttentionFusion(nn.Module):
    """
    CATF fusion: a standard Transformer encoder layer over the joint token sequence.

    Concatenates the token sequences of both branches, tags each token with a
    learnable branch embedding, prepends a learnable CLS token and applies
    pre-norm self-attention to the joint sequence. The fused CLS representation
    feeds the classifier.
    """

    def __init__(self, dim, n_classes, heads, layers, dropout, drop_path=0.0):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.mod_embed = nn.Parameter(torch.zeros(2, dim))   # per-branch tag
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mod_embed, std=0.02)

        # Flat drop-path rate on every fusion layer. (A linear 0->drop_path
        # schedule would leave a single-layer fusion at 0, disabling stochastic
        # depth exactly where we rely on it, so we apply the full rate to each.)
        self.encoder = nn.ModuleList(
            [FusionBlock(dim, heads, dropout, drop_path) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(dim // 2, n_classes),
        )
        self.capture = False         # when True, store the per-image branch attention share
        self._branch_share = None    # [B, 2]: CLS attention mass on (branch_a, branch_b)

    def forward(self, tok_a, tok_b):
        b = tok_a.size(0)
        na = tok_a.size(1)
        tok_a = tok_a + self.mod_embed[0]
        tok_b = tok_b + self.mod_embed[1]
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tok_a, tok_b], dim=1)         # [B, 1+Na+Nb, dim]
        for i, blk in enumerate(self.encoder):
            last = (i == len(self.encoder) - 1)
            if last and self.capture:
                seq, attw = blk(seq, return_attn=True)       # attw: [B, L, L]
                cls_attn = attw[:, 0, 1:]                     # CLS -> all tokens
                ra = cls_attn[:, :na].sum(1)
                rb = cls_attn[:, na:].sum(1)
                denom = (ra + rb).clamp_min(1e-6)
                self._branch_share = torch.stack([ra / denom, rb / denom], dim=1).detach()
            else:
                seq = blk(seq)
        return self.head(self.norm(seq[:, 0]))              # CLS token -> logits


class CrossBranchFusion(nn.Module):
    """
    Explicit cross-attention fusion variant (ablation).

    Unlike CrossAttentionFusion -- which applies *self*-attention over the
    concatenated token sequence -- here each branch is a separate query stream
    that cross-attends to the *other* branch (Q from one branch, K/V from the
    other, and vice-versa), so the "cross" in the name is literal. A learnable
    fusion query then cross-attends to the concatenation of both refined streams;
    the split of that query's attention mass across the two branches gives the
    per-image attention share of each branch.
    """

    def __init__(self, dim, n_classes, heads, layers, dropout, drop_path=0.0):
        super().__init__()
        self.mod_embed = nn.Parameter(torch.zeros(2, dim))
        self.q = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mod_embed, std=0.02)
        nn.init.trunc_normal_(self.q, std=0.02)

        self.na = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.nb = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.ab = nn.ModuleList(                      # branch A queries branch B
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
             for _ in range(layers)])
        self.ba = nn.ModuleList(                      # branch B queries branch A
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
             for _ in range(layers)])
        self.mlp_a = nn.ModuleList([self._mlp(dim, dropout) for _ in range(layers)])
        self.mlp_b = nn.ModuleList([self._mlp(dim, dropout) for _ in range(layers)])
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.fuse_q = nn.LayerNorm(dim)
        self.fuse_kv = nn.LayerNorm(dim)
        self.fuse = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(dim // 2, n_classes),
        )
        self.capture = False
        self._branch_share = None

    @staticmethod
    def _mlp(dim, dropout):
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(dim * 2, dim),
                             nn.Dropout(dropout))

    def forward(self, tok_a, tok_b):
        b = tok_a.size(0)
        na = tok_a.size(1)
        a = tok_a + self.mod_embed[0]
        bb = tok_b + self.mod_embed[1]
        for i in range(len(self.ab)):
            ha, hb = self.na[i](a), self.nb[i](bb)
            a = a + self.dp(self.ab[i](ha, hb, hb, need_weights=False)[0])
            bb = bb + self.dp(self.ba[i](hb, ha, ha, need_weights=False)[0])
            a = a + self.dp(self.mlp_a[i](a))
            bb = bb + self.dp(self.mlp_b[i](bb))
        kv = torch.cat([a, bb], dim=1)
        q = self.fuse_q(self.q.expand(b, -1, -1))
        k = self.fuse_kv(kv)
        fused, attw = self.fuse(q, k, k, need_weights=True,
                                average_attn_weights=True)    # attw: [B, 1, Na+Nb]
        if self.capture:
            ra = attw[:, 0, :na].sum(1)
            rb = attw[:, 0, na:].sum(1)
            denom = (ra + rb).clamp_min(1e-6)
            self._branch_share = torch.stack([ra / denom, rb / denom], dim=1).detach()
        return self.head(self.norm(fused.squeeze(1)))


class WFNet(nn.Module):
    """
    Waste Fusion Network: two backbones + CATF fusion.

    Flags expose the ablation variants:
      branches   : "both" | "a" (CNN-only) | "b" (Transformer-only)
      fusion_mode: "catf" (Transformer encoder over the joint tokens) | "concat"
      attn       : in-branch refinement, "cbam" | "se" | "none"
      use_cbam   : toggle; False -> attn="none", True -> config default
    """

    def __init__(self, num_classes, backbones=config.PROPOSED_BACKBONES,
                 dim=config.FUSION_DIM, pretrained=True,
                 branches="both", fusion_mode="catf", use_cbam=True,
                 attn=None, deep_supervision=None, catf_mode=None):
        super().__init__()
        self.branches = branches
        self.fusion_mode = fusion_mode
        # "self": standard self-attention over the concatenated tokens (default);
        # "cross": explicit bidirectional cross-attention (ablation variant).
        self.catf_mode = catf_mode or config.CATF_MODE
        # Resolve the in-branch attention: an explicit `attn` wins; otherwise
        # `use_cbam=False` maps to "none" and `use_cbam=True` to the project
        # default (config.BRANCH_ATTENTION).
        if attn is None:
            attn = config.BRANCH_ATTENTION if use_cbam else "none"
        self.attn_type = attn
        self.branch_a = BranchEncoder(backbones[0], dim, pretrained, attn) \
            if branches in ("both", "a") else None
        self.branch_b = BranchEncoder(backbones[1], dim, pretrained, attn) \
            if branches in ("both", "b") else None

        self.is_full = (fusion_mode == "catf" and branches == "both")
        if self.is_full:
            FusionCls = (CrossBranchFusion if self.catf_mode == "cross"
                         else CrossAttentionFusion)
            self.fusion = FusionCls(
                dim, num_classes, config.FUSION_HEADS,
                config.FUSION_LAYERS, config.FUSION_DROPOUT,
                drop_path=config.FUSION_DROP_PATH)
        else:  # concat / single-branch: simple dual-pool MLP head
            n_in = dim * (2 if branches == "both" else 1) * 2  # avg+max pooling
            self.head = nn.Sequential(
                nn.Dropout(config.FUSION_DROPOUT),
                nn.Linear(n_in, dim), nn.GELU(), nn.LayerNorm(dim),
                nn.Dropout(config.FUSION_DROPOUT / 2),
                nn.Linear(dim, num_classes))

        # Deep self-distillation: auxiliary per-branch heads used ONLY
        # during training (discarded at inference -> no extra cost).
        if deep_supervision is None:
            deep_supervision = config.DEEP_SUPERVISION
        self.deep_supervision = bool(deep_supervision) and self.is_full
        if self.deep_supervision:
            self.aux_a = nn.Linear(2 * dim, num_classes)
            self.aux_b = nn.Linear(2 * dim, num_classes)

    @staticmethod
    def _pool(tokens):                       # [B, N, D] -> [B, 2D]
        return torch.cat([tokens.mean(1), tokens.amax(1)], dim=1)

    def forward(self, x):
        ta = self.branch_a(x) if self.branch_a is not None else None
        tb = self.branch_b(x) if self.branch_b is not None else None
        if self.is_full:
            fused = self.fusion(ta, tb)
            if self.deep_supervision and self.training:
                # cache aux logits + teacher for the distillation loss
                self._aux_a = self.aux_a(self._pool(ta))
                self._aux_b = self.aux_b(self._pool(tb))
                self._fused_cache = fused
            return fused
        pooled = [self._pool(t) for t in (ta, tb) if t is not None]
        return self.head(torch.cat(pooled, dim=1))

    def distillation_loss(self, ce_fn):
        """
        Extra training loss = auxiliary supervision + self-distillation.
        Each branch is supervised directly and pulled towards the (detached)
        fused teacher's softened distribution. Zero cost at inference.

        ``ce_fn(logits)`` computes the (possibly MixUp/CutMix-mixed) supervised
        loss for a set of logits, so the aux heads stay consistent with the
        main head's targets.
        """
        T = config.KD_TEMPERATURE
        aux_ce = ce_fn(self._aux_a) + ce_fn(self._aux_b)
        teacher = F.softmax(self._fused_cache.detach() / T, dim=1)
        kd = 0.0
        for aux in (self._aux_a, self._aux_b):
            kd = kd + F.kl_div(F.log_softmax(aux / T, dim=1), teacher,
                               reduction="batchmean") * (T * T)
        return config.AUX_WEIGHT * aux_ce + config.KD_WEIGHT * kd

    def gradcam_target_layer(self):
        """Grad-CAM hooks the CNN branch's attention (interpretable texture map)."""
        b = self.branch_a if self.branch_a is not None else self.branch_b
        return b.attn


# Alias
ProposedModel = WFNet


# Ablation variants of the proposed model.
ABLATION_MODELS = {
    "Ablation: CNN-only":        dict(branches="a", fusion_mode="concat"),
    "Ablation: Transformer-only":dict(branches="b", fusion_mode="concat"),
    "Ablation: Dual (concat)":   dict(branches="both", fusion_mode="concat"),
    "Ablation: Dual+CATF w/o attn": dict(branches="both", fusion_mode="catf",
                                        attn="none"),
    "Ablation: WF-Net w/ CBAM":  dict(branches="both", fusion_mode="catf",
                                        attn="cbam"),   # SE (proposed) vs CBAM
    "Ablation: WF-Net w/o self-distill": dict(branches="both",
                                        fusion_mode="catf", deep_supervision=False),
    "Ablation: CATF cross-attention": dict(branches="both", fusion_mode="catf",
                                        catf_mode="cross"),  # explicit cross-attn
    # same fusion design on lightweight backbones
    "Ablation: WF-Net-Lite": dict(backbones=config.LITE_BACKBONES),
    # full model == WF-Net (Proposed): branches=both, catf(self), SE attn, self-distill
}


def build_model(display_name: str, num_classes: int) -> nn.Module:
    """Return a ready-to-train model for the given display name."""
    if display_name == config.PROPOSED_NAME:
        return WFNet(num_classes)
    if display_name in ABLATION_MODELS:
        return WFNet(num_classes, **ABLATION_MODELS[display_name])
    timm_name = config.BASELINE_MODELS[display_name]
    return timm.create_model(timm_name, pretrained=True,
                             num_classes=num_classes)


def split_param_groups(model: nn.Module):
    """Separate backbone vs. head params for discriminative learning rates."""
    if isinstance(model, WFNet):
        backbone, head = [], []
        for name, p in model.named_parameters():
            if ".backbone." in name:          # the two pretrained backbones
                backbone.append(p)
            else:                             # attention, proj, fusion, cls, head
                head.append(p)
        return backbone, head
    # timm baselines expose the classifier via get_classifier()
    head_module = model.get_classifier()
    head_ids = {id(p) for p in head_module.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = list(head_module.parameters())
    return backbone, head
