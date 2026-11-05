"""
Run-safety helpers for a multi-day experiment on a Windows workstation.

  * single-instance lock  : two copies writing the same result folder would
                            corrupt it (e.g. a double click, or a start-up
                            shortcut plus a manual start). The lock stores the
                            PID *and* the process creation time, so a stale lock
                            left by a power cut is recognised even if Windows
                            has reused the PID for another program.
  * keep-awake            : asks Windows not to sleep while the run is alive
                            (per-process request; released automatically at exit,
                            no system setting is changed).
  * free-disk check       : refuse to start when the results drive is nearly full.
"""
import atexit
import ctypes
import json
import os
import shutil
import sys
from datetime import datetime

_IS_WIN = sys.platform.startswith("win")


def _creation_time(pid):
    """Process creation time (FILETIME int) or None if the process is gone."""
    if not _IS_WIN:
        try:
            os.kill(pid, 0)
            return 0
        except OSError:
            return None
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:
            return None                                # not STILL_ACTIVE
        ft = [ctypes.c_ulonglong() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]):
            return None
        return int(ft[0].value)
    finally:
        k32.CloseHandle(h)


class RunLock:
    def __init__(self, folder):
        self.path = os.path.join(folder, ".running.lock")
        self.acquired = False

    def acquire(self):
        """Return (True, msg) if acquired, (False, msg) if another live run owns it."""
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    info = json.load(fh)
                ct = _creation_time(info["pid"])
                if ct is not None and ct == info.get("created"):
                    return False, (f"Another run is already active (PID {info['pid']}, "
                                   f"started {info.get('started', '?')}).")
            except Exception:
                pass                                   # unreadable -> treat as stale
        info = {"pid": os.getpid(), "created": _creation_time(os.getpid()),
                "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(info, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self.acquired = True
        atexit.register(self.release)
        return True, "lock acquired"

    def release(self):
        if self.acquired and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.acquired = False


def keep_awake():
    """Prevent system sleep while this process runs (display may still turn off)."""
    if not _IS_WIN:
        return False
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ok = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS))
    return bool(ok)


def free_gb(folder):
    probe = folder
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe).free / 2 ** 30
