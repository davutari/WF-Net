"""
Minimal Grad-CAM for explainability (Selvaraju et al., 2017).

Dependency-free (pure hooks) so it works on the proposed model without extra
packages. Produces heatmaps showing which image regions drove each prediction.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits[torch.arange(x.size(0)), class_idx].sum()
        self.model.zero_grad()
        score.backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)     # GAP of grads
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        # per-image min-max normalization
        for i in range(cam.shape[0]):
            c = cam[i]
            cam[i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
        return cam, class_idx.cpu().numpy()


def denormalize(img_tensor, mean, std):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)
