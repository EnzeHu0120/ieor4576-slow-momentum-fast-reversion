"""Run the original author's LSTM on the WRDS single-sequence pilot.

Architecture, Sharpe loss, batching, validation callback, prediction, and
hyperparameter grids are imported from the author's repository. A resumable
random-search controller replaces the obsolete private Keras Tuner loop.
"""

import argparse
import contextlib
import gc
import hashlib
import io
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bootstrap(device):
    """Set the project runtime before importing TensorFlow or Keras."""
    if os.environ.get("PAPER67_PANEL24_ENV_READY") == "1":
        return
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0" if device == "gpu" else "-1"
    env.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    env.setdefault("TF_NUM_INTEROP_THREADS", "2")
    library_paths = [str(Path(sysconfig.get_path("purelib")) / "nvidia/cusolver/lib"),
                     "/usr/lib/wsl/lib"]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    env["PAPER67_PANEL24_ENV_READY"] = "1"
    os.execve(sys.executable, [sys.executable, "-u", str(Path(sys.argv[0]).resolve()),
                              *sys.argv[1:]], env)


def patch_compatibility(repo, output):
    """Preserve original files and apply two explicit compatibility edits."""
    edits = [
        ("mom_trans/deep_momentum_network.py",
         "keras.optimizers.Adam(lr=learning_rate, clipnorm=max_gradient_norm)",
         "keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)"),
        ("mom_trans/model_inputs.py", "dtype=np.int)", "dtype=int)"),
    ]
    records = []
    for relative, old, new in edits:
        path = repo / relative
        original = path.read_text()
        before = digest(path)
        require(old in original or new in original,
                "Unrecognized author source; cannot safely patch " + relative)
        changed = old in original
        if changed:
            backup = output / "compat_original" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(original.replace(old, new))
        records.append({"file": relative, "changed_this_run": changed,
                        "old": old, "new": new,
                        "sha256_before": before, "sha256_after": digest(path)})
    save_json(output / "compatibility_last_run.json", records)
    return {row["file"]: row["sha256_after"] for row in records}

