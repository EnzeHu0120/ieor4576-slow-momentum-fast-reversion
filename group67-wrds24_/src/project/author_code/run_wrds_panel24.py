"""Prepare and run a fixed 24-series WRDS adaptation of the author's DMN.

Run from the original trading-momentum-transformer repository root.
All data policies and experimental differences are recorded in the output.
"""

import argparse
import contextlib
import gc
import hashlib
import io
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

from paper67_panel_runtime import bootstrap, digest, patch_compatibility, require, save_json

RAW = Path("data/raw/wrds_panel24")
PROCESSED = Path("data/processed/wrds_panel24")
RESULTS = Path("results/wrds_panel24")
BASE = ["norm_daily_return", "norm_monthly_return", "norm_quarterly_return",
        "norm_biannual_return", "norm_annual_return", "macd_8_24", "macd_16_48", "macd_32_96"]
LOOKBACKS = [10, 21, 63, 126, 252]
FEATURE_PATH = PROCESSED / "features.csv"
LEVEL_PATH = PROCESSED / "segmented_levels.csv"
CPD_PATH = PROCESSED / "features_cpd.csv"
POLICY = {
    "universe": "24 distinct provisional underlying candidates; selected by coverage, not performance",
    "raw_start": "2010-01-01", "raw_end": "2020-12-31",
    "training": "2010-2016 subject to per-asset feature warm-up and segment availability",
    "validation": "2017-2018", "test": "2019-2020",
    "reserved": "2021-2025 not read by feature construction or model evaluation",
    "price_changes": "No price is corrected, rescaled, interpolated, or forward/backward filled",
    "sugar_session_rule": "Use only dates also observed in both cocoa 8634 and coffee 9896; conservative shared-ICE-softs session restriction",
    "ICE_special_closure": "Exclude 2011-01-03 for sugar, cocoa, and coffee; documented ICE trading suspension",
    "feeder_cattle_exclusion": "Exclude 2012-04-06; source index drop contradicts CME agricultural prior-settlement convention on that date",
    "mexican_peso_break": "Start an independent history segment at 2017-01-13; do not assume or apply a 10x correction",
    "unexplained_gap_rule": "Start a new independent segment after a gap exceeding seven calendar days",
    "unexpected_large_move_rule": "Stop if a remaining adjacent raw index return exceeds 30% in magnitude",
    "sequence_rule": "Author batching, 63 observations, independent segments; no sequence crosses a history break",
    "warmup": "At least 253 index observations before the signal, identical for all five CPD lookbacks and ordinary LSTM",
    "training_purge": "A label whose signal and realized return fall in different calendar splits is excluded from fitting",
    "validation_portfolio": "Sum available contributions divided by fixed N=24 on union validation dates; unready assets are unallocated",
    "test_portfolio": "Fixed N=24; score only common dates with all assets and one identical signal/return interval",
    "test_calendar_limit": "Complete-interval comparison omits unmatched market days; it is not a full daily executable portfolio track record",
    "returns": "Primary author winsorized-index volatility-scaled targets; raw-index diagnostics also saved",
    "risk": "15% annual asset volatility target before dividing by 24; no portfolio rescaling, costs, or exposure cap",
    "paper_limit": "WRDS partial method replication; not Pinnacle 50-asset 1990-2020 numerical replication",
    "calendar_sources": [
        "https://www.ice.com/publicdocs/futures_us/exchange_notices/ExNot2012Holidays.pdf",
        "https://www.ice.com/publicdocs/futures_us/exchange_notices/ExNot2018HolidayCal.pdf",
        "https://www.cmegroup.com/tools-information/lookups/advisories/clearing/Chadv12-140.html",
        "https://www.cftc.gov/sites/default/files/stellent/groups/public/%40rulesandproducts/documents/ifdocs/rul121310iceus006.pdf",
    ],
}


def read_table(path):
    import pandas as pd
    x = pd.read_csv(path)
    for name in ["signal_date", "target_date"]:
        if name in x:
            x[name] = pd.to_datetime(x[name])
    return x


def freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(json.loads(path.read_text()) == value,
                "Existing configuration differs: " + str(path) + ". Use a new output directory.")
    else:
        save_json(path, value)


def construct_features(raw, check_prefix=True):
    import numpy as np
    import pandas as pd
    from mom_trans.data_prep import deep_momentum_strategy_features
    from mom_trans.classical_strategies import MACDStrategy

    dates = {
        int(c): set(g.Date_) for c, g in raw.groupby("CalcSeriesCode")
    }
    sugar_sessions = dates[8634] & dates[9896]
    parts, levels, actions, checks = [], [], [], []
    for code, group in raw.groupby("CalcSeriesCode", sort=True):
        code = int(code)
        group = group.sort_values("Date_").copy()
        exclude = pd.Series(False, index=group.index)
        if code == 2374:
            exclude = ~group.Date_.isin(sugar_sessions)
        if code in [2374, 8634, 9896]:
            exclude |= group.Date_.eq(pd.Timestamp("2011-01-03"))
        if code == 23163:
            exclude |= group.Date_.eq(pd.Timestamp("2012-04-06"))
        for _, r in group[exclude].iterrows():
            actions.append({"code": code, "date": str(r.Date_.date()),
                            "source_price": float(r.Settlement), "action": "exclude_observation",
                            "reason": ("ICE_documented_2011_special_suspension" if str(r.Date_.date()) == "2011-01-03"
                                       else "shared_ICE_softs_session_restriction" if code == 2374
                                       else "CME_2012_Good_Friday_settlement_review")})
        group = group[~exclude].copy()
        gap = group.Date_.diff().dt.days.gt(7)
        boundary = gap.copy()
        if code == 23253:
            boundary |= group.Date_.eq(pd.Timestamp("2017-01-13"))
        for idx in group.index[boundary]:
            actions.append({"code": code, "date": str(group.loc[idx, "Date_"].date()),
                            "source_price": float(group.loc[idx, "Settlement"]),
                            "action": "restart_independent_segment",
                            "reason": "unverified_price_scale_break" if code == 23253 and str(group.loc[idx, "Date_"].date()) == "2017-01-13" else "calendar_gap_above_seven_days"})
        group["segment_number"] = boundary.cumsum().astype(int)
        for number, g in group.groupby("segment_number", sort=True):
            segment = "WRDS_%d_SEG%02d" % (code, number)
            p = g.set_index("Date_")[["Settlement"]].rename(columns={"Settlement": "close"})
            require((p.close > 0).all() and np.isfinite(p.close).all(), "Invalid index: " + segment)
            r = p.close.pct_change(fill_method=None)
            require(not r.abs().gt(0.30).any(), "Unresolved >30% raw index move in " + segment + "; inspect source before training.")
            level = p.assign(code=code, segment=segment, position=np.arange(len(p)))
            levels.append(level.reset_index().rename(columns={"Date_": "signal_date"}))
            if len(p) < 256:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                f = deep_momentum_strategy_features(p)
            f["target_date"] = p.index.to_series().shift(-1).reindex(f.index)
            f["source_position"] = p.index.get_indexer(f.index)
            f = f[f.source_position.ge(253) & f.target_date.notna()].copy()
            if f.empty:
                continue
            if check_prefix:
                for cutoff in ["2016-12-30", "2018-12-31", "2020-12-31"]:
                    prefix_p = p.loc[:cutoff]
                    if len(prefix_p) < 256:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        prefix_f = deep_momentum_strategy_features(prefix_p)
                    common = f.index.intersection(prefix_f.index)
                    if len(common):
                        np.testing.assert_allclose(f.loc[common, BASE + ["daily_vol", "srs", "target_returns"]],
                                                   prefix_f.loc[common, BASE + ["daily_vol", "srs", "target_returns"]],
                                                   rtol=1e-11, atol=1e-12)
                        checks.append({"segment": segment, "cutoff": cutoff, "rows": len(common)})
            f["raw_next_index_return"] = r.shift(-1).reindex(f.index)
            f["raw_scaled_target"] = f.raw_next_index_return * 0.15 / (f.daily_vol * np.sqrt(252))
            f["macd_position"] = sum(MACDStrategy.scale_signal(f[n]) for n in BASE[-3:]) / 3
            f["code"], f["segment"] = code, segment
            f["signal_date"] = f.index
            f["split"] = np.where(f.target_date.lt("2017-01-01"), "train", np.where(f.target_date.lt("2019-01-01"), "valid", "test"))
            signal_split = np.where(f.signal_date.lt("2017-01-01"), "train", np.where(f.signal_date.lt("2019-01-01"), "valid", "test"))
            f["label_crosses_split"] = f["split"].to_numpy() != signal_split
            columns = ["code", "segment", "signal_date", "target_date", "source_position", "split", "label_crosses_split"] + BASE + ["daily_vol", "target_returns", "raw_next_index_return", "raw_scaled_target", "macd_position"]
            parts.append(f[columns].reset_index(drop=True))
    require(parts, "No usable author feature rows.")
    frame = pd.concat(parts, ignore_index=True).sort_values(["code", "signal_date"]).reset_index(drop=True)
    require(np.isfinite(frame[BASE + ["target_returns", "raw_scaled_target"]]).all().all(), "Non-finite features or labels.")
    require((frame.signal_date < frame.target_date).all(), "Noncausal dates.")
    require(not frame.duplicated(["code", "target_date"]).any(), "Duplicate feature keys.")
    return frame, pd.concat(levels, ignore_index=True), pd.DataFrame(actions), checks


def formatter(columns):
    from mom_trans.model_inputs import ModelFeatures, DataTypes, InputTypes
    obj = object.__new__(ModelFeatures)
    obj.total_time_steps = 63
    obj._column_definition = [
        ("segment", DataTypes.CATEGORICAL, InputTypes.ID),
        ("date", DataTypes.DATE, InputTypes.TIME),
        ("target_returns", DataTypes.REAL_VALUED, InputTypes.TARGET),
    ] + [(c, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT) for c in columns]
    return obj


def batches(frame, columns, test=False):
    import numpy as np
    import pandas as pd
    obj = formatter(columns)
    f = frame.copy().sort_values(["segment", "target_date"])
    f.index = pd.DatetimeIndex(f.target_date, name="return_date")
    f["date"] = f.index
    require(np.isfinite(f[columns + ["target_returns"]]).all().all(), "Non-finite model data.")
    if not test:
        data = []
        for split in ["train", "valid"]:
            g = f[f.split.eq(split) & ~f.label_crosses_split.astype(bool)]
            require(len(g) > 0, "Empty " + split + " split.")
            data.append(obj._batch_data(g, False))
        return data
    context = []
    for _, g in f.groupby("segment"):
        current = g[g.target_date.ge("2019-01-01") & g.target_date.lt("2021-01-01")]
        if len(current):
            previous = g[g.target_date.lt("2019-01-01")].tail(62)
            context.append(pd.concat([previous, current]))
    buff = pd.concat(context)
    counts = buff.groupby("segment").size()
    buff = buff[buff.segment.isin(counts[counts.ge(63)].index)]
    batch = obj._batch_data(buff, True)
    last = pd.to_datetime(batch["date"][:, -1, 0])
    keep = (last >= pd.Timestamp("2019-01-01")) & (last < pd.Timestamp("2021-01-01"))
    return {k: v[keep] for k, v in batch.items()}


def eligible_test(frame):
    import pandas as pd
    b = batches(frame, BASE, test=True)
    keys = pd.DataFrame({"segment": b["identifier"][:, -1, 0],
                         "target_date": pd.to_datetime(b["date"][:, -1, 0])})
    test = keys.merge(frame, on=["segment", "target_date"], validate="one_to_one")
    counts = test.groupby(["signal_date", "target_date"]).code.nunique()
    common = counts[counts.eq(24)].reset_index()[["signal_date", "target_date"]]
    result = test.merge(common, on=["signal_date", "target_date"], validate="many_to_one")
    require(result.code.nunique() == 24 and len(common) >= 400,
            "Fewer than 400 matching intervals or fewer than 24 evaluation series.")
    require(result.groupby("target_date").size().eq(24).all(), "Test panel is not balanced.")
    return result.sort_values(["target_date", "code"]).reset_index(drop=True)


def summarize_daily(daily):
    import pandas as pd
    from mom_trans.classical_strategies import calc_performance_metrics
    rows = []
    for name in daily:
        d = daily[[name]].rename(columns={name: "captured_returns"}).assign(identifier="PORTFOLIO_24")
        rows.append({"strategy": name, "common_intervals": len(d), **calc_performance_metrics(d)})
    return pd.DataFrame(rows)


def baseline(frame, output):
    import numpy as np
    x = eligible_test(frame)
    annual = np.sign(x.norm_annual_return)
    monthly = np.sign(x.norm_monthly_return)
    for name, pos in [("long_only", 1), ("tsmom_annual", annual),
                      ("tsmom_blend", 0.5 * annual + 0.5 * monthly),
                      ("tsmom_monthly", monthly), ("macd", x.macd_position)]:
        x[name + "_author_gross"] = pos * x.target_returns
        x[name + "_raw_gross"] = pos * x.raw_scaled_target
    cols = [c for c in x if c.endswith("_gross")]
    daily = x.groupby("target_date")[cols].sum() / 24.0
    daily.index.name = "return_date"
    x.to_csv(output / "baseline_asset_intervals.csv", index=False)
    daily.to_csv(output / "baseline_daily_2019_2020.csv")
    summary = summarize_daily(daily)
    summary.to_csv(output / "baseline_metrics_2019_2020.csv", index=False)
    return x, summary


def prepare(args):
    import numpy as np
    import pandas as pd
    repo = Path.cwd()
    require((repo / "mom_trans/data_prep.py").is_file(), "Run from the author's repository root.")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "preparation"
    output.mkdir(parents=True, exist_ok=True)
    patch_compatibility(repo, output)
    universe = json.loads((RAW / "universe24.json").read_text())
    raw = pd.read_csv(RAW / "source_daily.csv", dtype={"CalcSeriesCode": str})
    raw["CalcSeriesCode"] = raw.CalcSeriesCode.astype(int)
    raw["Date_"] = pd.to_datetime(raw.Date_)
    raw = raw[raw.Date_.between("2010-01-01", "2020-12-31")].copy()
    expected = {int(a["code"]) for a in universe["assets"]}
    require(set(raw.CalcSeriesCode.unique()) == expected and len(expected) == 24, "Universe mismatch.")
    require(not raw.duplicated(["CalcSeriesCode", "Date_"]).any(), "Duplicate raw daily keys.")
    cfg = {"policy": POLICY, "raw_sha256": digest(RAW / "source_daily.csv"),
           "universe_sha256": digest(RAW / "universe24.json"),
           "author_feature_sha256": digest(repo / "mom_trans/data_prep.py"),
           "author_classical_sha256": digest(repo / "mom_trans/classical_strategies.py"),
           "runner_sha256": digest(Path(__file__)),
           "pandas": pd.__version__, "numpy": np.__version__}
    freeze(output / "preparation_config.json", cfg)
    frame, levels, actions, checks = construct_features(raw)
    require(set(frame.code.unique()) == expected, "An asset has no usable features.")
    frame.to_csv(FEATURE_PATH, index=False)
    levels.to_csv(LEVEL_PATH, index=False)
    actions.to_csv(output / "data_actions.csv", index=False)
    coverage = frame.groupby(["code", "split"]).size().unstack(fill_value=0)
    coverage.to_csv(output / "feature_coverage.csv")
    require((coverage[["train", "test"]] > 0).all().all(), "An asset has no training or test rows.")
    # A post-break segment may enter validation without its own training data;
    # the shared network has no ticker/segment embedding or per-asset weights.
    selected, summary = baseline(frame, output)
    train, valid = batches(frame, BASE)
    audit = {"assets": 24, "feature_rows": len(frame), "history_segments": int(levels.segment.nunique()),
             "actions": len(actions), "causal_prefix_checks": checks,
             "train_shape": list(train["inputs"].shape), "valid_shape": list(valid["inputs"].shape),
             "train_labels": int(train["active_entries"].sum()),
             "valid_labels": int(valid["active_entries"].sum()),
             "common_test_intervals": int(selected.target_date.nunique()),
             "common_test_asset_rows": len(selected),
             "training_min_date": min(train["date"][train["date"] != ""]),
             "training_max_date": max(train["date"][train["date"] != ""]),
             "validation_min_date": min(valid["date"][valid["date"] != ""]),
             "validation_max_date": max(valid["date"][valid["date"] != ""]),
             "holdout_read_by_feature_builder": False, "cpd_fit_performed": False,
             "model_training_performed": False}
    save_json(output / "PREPARATION_COMPLETE.json", audit)
    print("Universe: 24 provisional WRDS series")
    print("Features:", len(frame), "| Independent histories:", audit["history_segments"])
    print("Training/validation sequence shapes:", train["inputs"].shape, valid["inputs"].shape)
    print("Common test intervals:", audit["common_test_intervals"], "| Assets per interval: 24")
    print("Causal prefix checks:", len(checks), "passed")
    print(summary[summary.strategy.str.endswith("author_gross")].to_string(index=False))
    print("PANEL24_PREPARATION_OK", flush=True)


def gp_task(task):
    from paper67_panel_gp import fit_window
    segment, window = task
    result = fit_window(window)
    result["segment"] = segment
    return result


def compute_cpd(args):
    for n in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[n] = "1"
    import numpy as np
    import pandas as pd
    from paper67_panel_gp import initialize_worker
    require(args.workers >= 1 and args.limit >= 0, "Invalid worker count or smoke limit.")
    features = read_table(FEATURE_PATH)
    levels = pd.read_csv(LEVEL_PATH, parse_dates=["signal_date"])
    cache = PROCESSED / "cpd_cache"
    output = RESULTS / "cpd"
    output.mkdir(parents=True, exist_ok=True)
    cfg = {"features_sha256": digest(FEATURE_PATH), "levels_sha256": digest(LEVEL_PATH),
           "author_cpd_sha256": digest(Path("mom_trans/changepoint_detection.py")),
           "lookbacks": LOOKBACKS, "iteration_cap": 200,
           "initialization": "author defaults independently reset each window",
           "raw_return_policy": "unwinsorized percent changes within each cleaned independent segment"}
    freeze(cache / "config.json", cfg)
    level_groups = {s: g.sort_values("position") for s, g in levels.groupby("segment")}
    task_info, smoke_keys = [], set()
    for (segment, code), g in features.groupby(["segment", "code"], sort=True):
        level = level_groups[segment]
        returns = level.close.pct_change(fill_method=None).to_numpy()
        for _, row in g.iterrows():
            pos = int(row.source_position)
            for lbw in LOOKBACKS:
                key = (int(code), lbw)
                if args.limit:
                    count = sum(1 for a, l, _ in smoke_keys if (a, l) == key)
                    if count >= args.limit:
                        continue
                    smoke_keys.add((int(code), lbw, str(row.signal_date.date())))
                path = cache / segment / ("lbw_%d" % lbw) / (str(row.signal_date.date()) + ".json")
                values = returns[pos - lbw: pos + 1]
                require(len(values) == lbw + 1 and np.isfinite(values).all(), "Invalid GP window.")
                task = (segment, (lbw, pos, str(row.signal_date.date()), str(level.signal_date.iloc[pos - lbw].date()), values.tolist()))
                task_info.append((path, task))
    paths = { (t[0], t[1][0], t[1][2]): p for p,t in task_info }
    pending = []
    records = {}
    for path, task in task_info:
        if path.exists():
            row = json.loads(path.read_text())
            require(row["segment"] == task[0] and row["lookback"] == task[1][0]
                    and row["date"] == task[1][2] and row["t"] == task[1][1], "Mismatched GP cache.")
            if row["status"] == "failed" and args.retry_failed:
                pending.append(task)
            else:
                records[(row["segment"], row["lookback"], row["date"])] = row
        else:
            pending.append(task)
    print("GP windows:", len(task_info), "| Cached:", len(records), "| To fit:", len(pending), flush=True)
    started = time.monotonic()
    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=initialize_worker,
                      initargs=(str(Path.cwd()),), maxtasksperchild=500) as pool:
            for i, row in enumerate(pool.imap_unordered(gp_task, pending, chunksize=1), 1):
                key = (row["segment"], row["lookback"], row["date"])
                save_json(paths[key], row)
                records[key] = row
                if i == 1 or i % 100 == 0 or i == len(pending):
                    print("GP %d/%d | %s | lbw %d | %s | elapsed %.1fs" %
                          (i, len(pending), row["segment"], row["lookback"], row["status"], time.monotonic() - started), flush=True)
    diagnostic = pd.DataFrame([{k:r.get(k) for k in ["segment", "lookback", "date", "status", "elapsed_seconds"]} for r in records.values()])
    diagnostic.to_csv(output / ("smoke_windows.csv" if args.limit else "fit_windows.csv"), index=False)
    if args.limit:
        require(len(records) == 24 * len(LOOKBACKS) * args.limit, "Incomplete 24-asset GP smoke test.")
        require(diagnostic.status.eq("completed").all(), "A GP smoke fit failed. Inspect smoke_windows.csv.")
        print("Mean fit seconds:", round(float(diagnostic.elapsed_seconds.mean()), 4))
        print("Smoke only: no complete CPD model feature table was emitted.")
        print("PANEL24_CPD_SMOKE_OK", flush=True)
        return
    full = features.copy()
    filled = []
    for segment, group in full.groupby("segment", sort=True):
        for lbw in LOOKBACKS:
            past = None
            for idx, row in group.sort_values("signal_date").iterrows():
                key = (segment, lbw, str(row.signal_date.date()))
                fit = records[key]
                if fit["status"] == "completed":
                    past = fit
                require(past is not None, "Initial GP failure has no prior fit: " + str(key) + ". Retry with --retry-failed.")
                source_date = pd.Timestamp(past["date"])
                require(source_date <= row.signal_date, "GP source is from the future.")
                full.loc[idx, "cp_score_%d" % lbw] = past["cp_score"]
                full.loc[idx, "cp_rl_%d" % lbw] = (int(row.source_position) - past["cp_location"]) / lbw
                full.loc[idx, "cp_source_date_%d" % lbw] = str(source_date.date())
                if fit["status"] != "completed":
                    filled.append({"segment": segment, "lookback": lbw, "date": fit["date"], "source_date": past["date"]})
    require(np.isfinite(full[["cp_score_%d" % l for l in LOOKBACKS] + ["cp_rl_%d" % l for l in LOOKBACKS]]).all().all(), "Missing CPD model inputs.")
    full.to_csv(CPD_PATH, index=False)
    save_json(output / "CPD_COMPLETE.json", {"windows": len(records), "failed_fits_filled_from_past": filled,
                                            "elapsed_seconds_this_invocation": time.monotonic() - started,
                                            "cpd_features_sha256": digest(CPD_PATH), "config": cfg})
    print("PANEL24_CPD_FEATURES_OK", flush=True)


def train(args):
    bootstrap(args.device)
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import keras_tuner as kt
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    patch_compatibility(Path.cwd(), output)
    from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel, SharpeValidationLoss
    from settings.hp_grid import (HP_HIDDEN_LAYER_SIZE, HP_DROPOUT_RATE, HP_MAX_GRADIENT_NORM,
                                  HP_LEARNING_RATE, HP_MINIBATCH_SIZE)
    uses_cpd = args.model in ["cpd", "all"]
    if uses_cpd:
        complete_path = RESULTS / "cpd/CPD_COMPLETE.json"
        require(complete_path.is_file() and CPD_PATH.is_file(), "Complete CPD feature outputs are missing. A GP smoke test does not produce them.")
        complete = json.loads(complete_path.read_text())
        require(complete["cpd_features_sha256"] == digest(CPD_PATH)
                and complete["config"]["features_sha256"] == digest(FEATURE_PATH)
                and complete["config"]["levels_sha256"] == digest(LEVEL_PATH), "CPD inputs or source feature hashes differ.")
    frame = read_table(CPD_PATH if uses_cpd else FEATURE_PATH)
    require(frame.code.nunique() == 24, "The training panel must contain exactly 24 assets.")
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu":
        require(gpus, "TensorFlow did not detect the GPU.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    device = "/GPU:0" if args.device == "gpu" else "/CPU:0"
    names = ["hidden_layer_size", "dropout_rate", "max_gradient_norm", "learning_rate", "batch_size"]
    grid = list(itertools.product(HP_HIDDEN_LAYER_SIZE, HP_DROPOUT_RATE, HP_MAX_GRADIENT_NORM,
                                  HP_LEARNING_RATE, HP_MINIBATCH_SIZE))
    require(1 <= args.trials <= len(grid) and args.epochs > 0 and args.patience > 0, "Invalid search budget.")
    indices = np.random.default_rng(args.seed).permutation(len(grid))[:args.trials]
    candidates = [dict(zip(names, grid[int(i)])) for i in indices]
    variants = ([None] if args.model in ["lstm", "all"] else []) + (LOOKBACKS if args.model in ["cpd", "all"] else [])
    cfg = {"seed": args.seed, "trials_per_variant": args.trials, "epochs": args.epochs,
           "patience": args.patience, "device": device, "model": args.model,
           "lookbacks": LOOKBACKS, "feature_sha256": digest(CPD_PATH if args.model in ["cpd", "all"] else FEATURE_PATH),
           "preparation_sha256": digest(RESULTS / "preparation/preparation_config.json"),
           "baseline_sha256": digest(RESULTS / "preparation/baseline_asset_intervals.csv"),
           "frozen_lstm_config_sha256": digest(Path(args.frozen_lstm) / "run_config.json") if args.frozen_lstm else None,
           "candidates": candidates, "validation_only": args.validation_only,
           "policy": POLICY, "tensorflow": tf.__version__, "python": platform.python_version(),
           "budget_is_smoke": args.trials != 50 or args.epochs != 300 or args.patience != 25,
           "author_model_sha256": digest(Path("mom_trans/deep_momentum_network.py")),
           "author_batch_sha256": digest(Path("mom_trans/model_inputs.py")),
           "runner_sha256": digest(Path(__file__))}
    freeze(output / "run_config.json", cfg)
    if args.frozen_lstm:
        require(args.model == "cpd", "--frozen-lstm applies only to --model cpd.")
        previous = json.loads((Path(args.frozen_lstm) / "run_config.json").read_text())
        require((Path(args.frozen_lstm) / "RUN_COMPLETE.json").is_file(), "The frozen ordinary LSTM run is not complete.")
        for key in ["seed", "trials_per_variant", "epochs", "patience", "candidates", "policy", "preparation_sha256", "baseline_sha256"]:
            require(previous[key] == cfg[key], "Frozen LSTM search or data differs: " + key)
        require(previous["feature_sha256"] == digest(FEATURE_PATH), "Frozen LSTM feature data differ.")
    all_statuses = []
    selected = []
    for lbw in variants:
        variant = "lstm" if lbw is None else "cpd_%d" % lbw
        columns = BASE if lbw is None else BASE + ["cp_score_%d" % lbw, "cp_rl_%d" % lbw]
        training, validation = batches(frame, columns)
        network = object.__new__(LstmDeepMomentumNetworkModel)
        network.time_steps, network.input_size, network.output_size = 63, len(columns), 1
        network.n_multiprocessing_workers = 1
        time_indices, num_time = network._index_times(validation["date"])
        # Convert the author's per-date mean into sum / fixed 24 without
        # changing its callback, Sharpe statistic, checkpointing, or patience.
        counts = np.bincount(time_indices.ravel(), minlength=num_time)
        counts[0] = 0
        validation_returns = validation["outputs"] * counts[time_indices] / 24.0

        def build(hp_values, seed):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            hp = kt.HyperParameters()
            for key in names[:-1]:
                hp.Fixed(key, hp_values[key])
            with tf.device(device):
                model = network.model_builder(hp)
            require(np.isclose(float(model.optimizer.learning_rate.numpy()), hp_values["learning_rate"]), "Learning-rate mismatch.")
            if args.device == "gpu":
                require("GPU" in model.weights[0].device, "Variables are not on GPU.")
            return model

        class AuditValidation(SharpeValidationLoss):
            def __init__(self, checkpoint, history, label):
                super().__init__(validation["inputs"], validation_returns, time_indices, num_time,
                                 args.patience, 1, weights_save_location=str(checkpoint))
                self.history_path, self.label = history, label
                self.records, self.best_epoch, self.best_steps = [], 0, 0

            def on_epoch_end(self, epoch, logs=None):
                if logs is None or not np.isfinite(logs["loss"]):
                    raise FloatingPointError("Non-finite training loss.")
                previous = self.best_sharpe
                steps = int(self.model.optimizer.iterations.numpy())
                with contextlib.redirect_stdout(io.StringIO()):
                    super().on_epoch_end(epoch, logs)
                if not np.isfinite(logs["sharpe"]):
                    raise FloatingPointError("Non-finite validation Sharpe.")
                if self.best_sharpe > previous:
                    self.best_epoch, self.best_steps = epoch + 1, steps
                self.records.append({"epoch": epoch + 1, "loss": float(logs["loss"]),
                                     "validation_sharpe": float(logs["sharpe"]), "optimizer_steps": steps})
                pd.DataFrame(self.records).to_csv(self.history_path, index=False)
                if epoch == 0 or (epoch + 1) % 5 == 0 or self.model.stop_training:
                    print("%s | epoch %d/%d | loss %.5f | validation %.5f | best %.5f | steps %d" %
                          (self.label, epoch + 1, args.epochs, logs["loss"], logs["sharpe"], self.best_sharpe, steps), flush=True)

        statuses = []
        for trial, hp_values in enumerate(candidates, 1):
            td = output / variant / ("trial_%03d" % trial)
            td.mkdir(parents=True, exist_ok=True)
            status_path = td / "status.json"
            checkpoint = td / "best.weights.h5"
            if status_path.exists():
                row = json.loads(status_path.read_text())
                require(row["parameters"] == hp_values, "Saved trial parameters differ.")
                require(row["status"] != "completed" or checkpoint.is_file(), "Saved checkpoint missing.")
                statuses.append(row)
                continue
            seed = args.seed + 1009 * trial
            model = build(hp_values, seed)
            label = "%s trial %d/%d" % (variant, trial, args.trials)
            print("Starting", label, hp_values, "| device", model.weights[0].device, flush=True)
            callback = AuditValidation(checkpoint, td / "history.csv", label)
            try:
                initial = model.predict(validation["inputs"], verbose=0) * validation_returns
                initial_daily = np.bincount(time_indices.ravel(), weights=initial.ravel(), minlength=num_time)[1:] / counts[1:]
                initial_sharpe = float(initial_daily.mean() / np.sqrt(initial_daily.var() + 1e-9) * np.sqrt(252))
                if not np.isfinite(initial_sharpe):
                    raise FloatingPointError("Non-finite untrained validation diagnostic.")
                save_json(td / "initial_validation.json", {"untrained_validation_sharpe_diagnostic": initial_sharpe,
                                                           "eligible_for_model_selection": False})
                model.fit(training["inputs"], training["outputs"], sample_weight=training["active_entries"],
                          batch_size=hp_values["batch_size"], epochs=args.epochs, callbacks=[callback],
                          shuffle=True, verbose=0)
                require(checkpoint.is_file(), "No finite validation checkpoint.")
                row = {"variant": variant, "lookback": lbw, "trial": trial, "seed": seed,
                       "parameters": hp_values, "status": "completed", "validation_sharpe": float(callback.best_sharpe),
                       "best_epoch": callback.best_epoch, "optimizer_steps_in_best_checkpoint": callback.best_steps,
                       "epochs_ran": len(callback.records), "checkpoint": str(checkpoint),
                       "training_sequences": int(len(training["inputs"])),
                       "batches_per_epoch": int(math.ceil(len(training["inputs"]) / hp_values["batch_size"]))}
            except (FloatingPointError, tf.errors.InvalidArgumentError) as error:
                row = {"variant": variant, "lookback": lbw, "trial": trial, "seed": seed,
                       "parameters": hp_values, "status": "failed_nonfinite", "validation_sharpe": None,
                       "error": str(error)[-1000:]}
            save_json(status_path, row)
            statuses.append(row)
            pd.DataFrame(statuses).to_csv(output / variant / "search_results.csv", index=False)
            del model, callback
            gc.collect()
        good = [r for r in statuses if r["status"] == "completed"]
        require(good, "All trials failed for " + variant)
        winner = max(good, key=lambda r: r["validation_sharpe"])
        selected.append(winner)
        all_statuses.extend(statuses)
        save_json(output / variant / "best_validation.json", winner)
    winners = {}
    plain = [r for r in selected if r["lookback"] is None]
    cp = [r for r in selected if r["lookback"] is not None]
    if plain:
        winners["lstm"] = plain[0]
    if cp:
        winners["lstm_cpd"] = max(cp, key=lambda r: r["validation_sharpe"])
    save_json(output / "SELECTED_USING_VALIDATION_ONLY.json", winners)
    if args.validation_only:
        print("No test predictions or test metrics were computed.")
        print("PANEL24_TRAINING_CHECK_OK", flush=True)
        return
    comparison = pd.read_csv(RESULTS / "preparation/baseline_asset_intervals.csv", parse_dates=["signal_date", "target_date"])
    if args.frozen_lstm:
        frozen = pd.read_csv(Path(args.frozen_lstm) / "asset_comparison_2019_2020.csv", parse_dates=["target_date"])
        comparison = comparison.merge(frozen[["segment", "target_date", "lstm_position"]],
                                      on=["segment", "target_date"], how="left", validate="one_to_one")
        require(comparison.lstm_position.notna().all(), "Frozen LSTM predictions do not match common test intervals.")
        comparison["lstm_author_gross"] = comparison.lstm_position * comparison.target_returns
        comparison["lstm_raw_gross"] = comparison.lstm_position * comparison.raw_scaled_target
    for name, best in winners.items():
        lbw = best["lookback"]
        columns = BASE if lbw is None else BASE + ["cp_score_%d" % lbw, "cp_rl_%d" % lbw]
        network.input_size = len(columns)
        model = build(best["parameters"], best["seed"])
        model.optimizer.build(model.trainable_variables)
        model.load_weights(best["checkpoint"])
        batch = batches(frame, columns, test=True)
        positions = model.predict(batch["inputs"], verbose=0)[:, -1, 0]
        require(np.isfinite(positions).all() and (np.abs(positions) <= 1 + 1e-6).all(), "Invalid predicted positions.")
        # Saved checkpoint replay is checked on the same runtime and backend.
        replay = model.predict(batch["inputs"], verbose=0)[:, -1, 0]
        np.testing.assert_allclose(positions, replay, rtol=1e-6, atol=1e-7)
        pred = pd.DataFrame({"segment": batch["identifier"][:, -1, 0],
                             "target_date": pd.to_datetime(batch["date"][:, -1, 0]),
                             name + "_position": positions})
        comparison = comparison.merge(pred, on=["segment", "target_date"], how="left", validate="one_to_one")
        require(comparison[name + "_position"].notna().all(), "A common test prediction is missing.")
        comparison[name + "_author_gross"] = comparison[name + "_position"] * comparison.target_returns
        comparison[name + "_raw_gross"] = comparison[name + "_position"] * comparison.raw_scaled_target
        model.save_weights(str(output / ("best_%s.weights.h5" % name)))
        del model
        gc.collect()
    comparison.to_csv(output / "asset_comparison_2019_2020.csv", index=False)
    cols = [c for c in comparison if c.endswith("_gross")]
    daily = comparison.groupby("target_date")[cols].sum() / 24
    daily.index.name = "return_date"
    daily.to_csv(output / "daily_comparison_2019_2020.csv")
    summary = summarize_daily(daily)
    summary.to_csv(output / "comparison_2019_2020.csv", index=False)
    save_json(output / "RUN_COMPLETE.json", {"attempted_trials": len(all_statuses),
                                           "completed_trials": sum(r["status"] == "completed" for r in all_statuses),
                                           "common_intervals": len(daily), "assets": 24, "device": device,
                                           "budget_is_smoke": cfg["budget_is_smoke"], "policy": POLICY,
                                           "gpu_details": [tf.config.experimental.get_device_details(g) for g in gpus]})
    bundle = output / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p != bundle:
                z.write(p, p.relative_to(output))
        for p in [RESULTS / "preparation/PREPARATION_COMPLETE.json", RESULTS / "preparation/preparation_config.json",
                  RESULTS / "preparation/data_actions.csv", RAW / "universe24.json", Path(__file__)]:
            z.write(p, Path("audit_inputs") / p.name)
        if args.frozen_lstm:
            for name in ["best_lstm.weights.h5", "run_config.json", "SELECTED_USING_VALIDATION_ONLY.json",
                         "RUN_COMPLETE.json", "asset_comparison_2019_2020.csv"]:
                z.write(Path(args.frozen_lstm) / name, Path("frozen_lstm") / name)
    print(summary[summary.strategy.str.endswith("author_gross")].to_string(index=False))
    print("Results bundle:", bundle)
    print("PANEL24_TRAINING_RUN_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Build causal segmented features, audit data actions, and run traditional baselines.")
    gp = sub.add_parser("cpd", help="Fit original GPflow CPD features, with persistent per-window caches.")
    gp.add_argument("--workers", type=int, default=2)
    gp.add_argument("--limit", type=int, default=0, help="Smoke only: dates per asset per lookback; emits no full feature table.")
    gp.add_argument("--retry-failed", action="store_true")
    tr = sub.add_parser("train", help="Tune shared original-author LSTMs; optionally include all five CPD lookbacks.")
    tr.add_argument("--model", choices=["lstm", "cpd", "all"], default="lstm")
    tr.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    tr.add_argument("--trials", type=int, default=50, help="Trials for ordinary LSTM and for EACH CPD lookback.")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--patience", type=int, default=25)
    tr.add_argument("--seed", type=int, default=67)
    tr.add_argument("--output", default="results/wrds_panel24/lstm_seed67")
    tr.add_argument("--validation-only", action="store_true")
    tr.add_argument("--frozen-lstm", default="", help="Completed matched ordinary LSTM output to include without retraining; only for --model cpd.")
    args = parser.parse_args()
    require(Path("mom_trans/data_prep.py").is_file(), "Run from ~/paper67-author.")
    sys.path.insert(0, str(Path.cwd()))
    if args.command == "prepare":
        prepare(args)
    elif args.command == "cpd":
        compute_cpd(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
