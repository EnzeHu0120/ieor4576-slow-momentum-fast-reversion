"""Compute causal CPD features using the original author's GPflow function."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path


_FIT = None
_WORKER_ERROR = None
_NP = None
_PD = None
_WORKER_COUNT = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize_worker(repo):
    global _FIT, _WORKER_ERROR, _NP, _PD
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        sys.path.insert(0, repo)
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
            import mom_trans.changepoint_detection as cpd
        tf.get_logger().setLevel("ERROR")
        require(cpd.MAX_ITERATIONS == 200, "The expected original GP iteration cap changed.")
        _NP, _PD, _FIT = np, pd, cpd.changepoint_loc_and_score
    except Exception:
        # Return initialization failures as failed tasks instead of allowing
        # multiprocessing.Pool to endlessly respawn failing initializers.
        _WORKER_ERROR = traceback.format_exc()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


def fit_window(task):
    global _WORKER_COUNT
    lbw, position, date, start_date, returns = task
    result = {"lookback": lbw, "t": position, "date": date,
              "window_start_date": start_date, "window_end_date": date,
              "observations": lbw + 1}
    started = time.monotonic()
    try:
        if _WORKER_ERROR:
            raise RuntimeError(_WORKER_ERROR)
        window = _PD.DataFrame({
            "X": _NP.arange(position - lbw, position + 1, dtype=float),
            "Y": _NP.asarray(returns, dtype=float),
        })
        score, location, normalized, matern, change = _FIT(window)
        values = [float(score), float(location), float(normalized)]
        require(_NP.isfinite(values).all(), "Non-finite CPD output.")
        require(-1e-8 <= values[0] <= 1 + 1e-8 and
                -1e-8 <= values[2] <= 1 + 1e-8, "CPD output outside its window bounds.")
        require(abs((position - values[1]) / lbw - values[2]) < 1e-8,
                "CPD location normalization does not match the original convention.")
        result.update({"status": "completed", "cp_score": values[0],
                       "cp_location": values[1], "cp_location_norm": values[2],
                       "matern_parameters": json_safe(matern),
                       "changepoint_parameters": json_safe(change)})
        json.dumps(result, allow_nan=False)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # The original function raises BaseException for a failed default
        # initialization, so Exception alone would not capture that failure.
        result = {key: result[key] for key in ["lookback", "t", "date",
                                              "window_start_date", "window_end_date", "observations"]}
        result.update({"status": "failed", "error": traceback.format_exc()[-3000:]})
    result["elapsed_seconds"] = time.monotonic() - started
    _WORKER_COUNT += 1
    if _WORKER_COUNT % 20 == 0:
        gc.collect()
    return result

