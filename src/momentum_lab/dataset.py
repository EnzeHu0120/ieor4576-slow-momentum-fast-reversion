"""Causal calendar rolls and the authors' feature construction on WRDS histories."""
import importlib.util
import ast
import json
import sys
import types
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from .paths import ROOT, configuration, prepare_directories

FEATURE_COLUMNS = ["norm_daily_return", "norm_monthly_return", "norm_quarterly_return",
    "norm_biannual_return", "norm_annual_return", "macd_8_24", "macd_16_48", "macd_32_96"]

def audit_held_prices():
    """Reject unsupported percentage P&L rather than silently dropping a real loss."""
    rows=[]
    for path in sorted((ROOT/'data/interim').glob('*_rolls.parquet')):
        f=pd.read_parquet(path,columns=['held_contract','held_price','prior_price','stale_mark','invalid_return'])
        held=f.held_contract.notna()&f.prior_price.gt(0)
        rows.append({'ticker':path.stem.removesuffix('_rolls'),
            'held_negative_price_days':int((held&f.held_price.lt(0)).sum()),
            'held_zero_price_days':int((held&f.held_price.eq(0)).sum()),
            'stale_marks':int(f.stale_mark.sum()),'invalid_held_return_days':int((held&f.invalid_return).sum())})
    audit=pd.DataFrame(rows)
    audit.to_csv(ROOT/'results/replication/tables/held_price_sanity.csv',index=False)
    if audit.invalid_held_return_days.sum():
        raise ValueError('Held-contract percentage returns are invalid; audit dollar P&L before fitting models')
    return audit

def author_features():
    """Execute preserved author source with only pandas API compatibility changes.

    Source and MIT license remain unchanged under vendor/. Backfills in the author
    code are restricted to warmup and removed by the 314-observation feature burnin.
    """
    base = ROOT / "vendor/author-original/mom_trans"
    pkg = types.ModuleType("mom_trans"); pkg.__path__ = [str(base)]
    sys.modules["mom_trans"] = pkg
    for name in ["classical_strategies", "data_prep"]:
        module = types.ModuleType("mom_trans." + name)
        module.__file__ = str(base / (name + ".py"))
        source = (base / (name + ".py")).read_text()
        source = source.replace('.fillna(method="bfill")', '.bfill()')
        source = source.replace('.fillna(method="ffill")', '.ffill()')
        source = source.replace('.index.weekofyear', '.index.isocalendar().week.to_numpy()')
        tree=ast.parse(source)
        # Feature functions never call empyrical. Omit its unused import, avoiding
        # optional market-data network initialization in modern empyrical packages.
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=="empyrical")]
        exec(compile(tree, module.__file__, "exec"), module.__dict__)
        sys.modules[module.__name__] = module
    return sys.modules["mom_trans.data_prep"].deep_momentum_strategy_features

def rolled_returns(bars, metadata, roll_days=5):
    """Choose at close t, earn the same held contract's t-to-t+1 price change.

    No cross-maturity price change is ever booked. Notice and last-trade dates are
    vendor fields; no expiry inferred from final observed bar is used.
    """
    bars = bars.copy(); metadata = metadata.copy()
    bars["date"] = pd.to_datetime(bars.date)
    bars["futcode"] = bars.futcode.astype(int)
    # Conflicting duplicates must not be silently selected.
    bars = bars.drop_duplicates()
    if bars.duplicated(["date", "futcode"]).any():
        raise ValueError("Conflicting settlements for a contract/date")
    metadata["futcode"] = metadata.futcode.astype(int)
    m = metadata.drop_duplicates("futcode").set_index("futcode")
    safe = pd.concat([pd.to_datetime(m.lasttrddate), pd.to_datetime(m.firstnoticedate)],axis=1).min(axis=1)
    safe = safe.dropna().map(lambda d: d - BDay(roll_days))
    p = bars.pivot(index="date", columns="futcode", values="settlement").sort_index().astype(float)
    # Remove rows on which this market has no actual observation at all.
    p = p.dropna(how="all")
    oi = bars.pivot(index="date", columns="futcode", values="oi").reindex_like(p).astype(float)
    volume = bars.pivot(index="date", columns="futcode", values="volume").reindex_like(p).astype(float)
    codes = [x for x in p if x in safe.index]
    p, oi, volume = p[codes], oi[codes], volume[codes]
    # Appendix A fixes the market universe. Historical volume/OI fields are often
    # absent even when settlements exist; their absence is not market inception.
    # Select using prices observed now and dates known before trading.
    eligible = p.notna() & p.gt(0)
    for code in codes:
        eligible[code] &= p.index < safe[code]
    ranked = sorted(codes, key=lambda x:(safe[x], x))
    choice = eligible[ranked].idxmax(axis=1).where(eligible.any(axis=1))
    held = choice.shift(1)
    # A missing settlement is valued at the last observed mark, causally. Do not
    # remove a position based on knowledge that tomorrow's quote will be missing.
    # The next available quote realizes its full change. Flag every stale mark.
    marked=p.ffill()
    within = marked / marked.shift(1) - 1
    within = within.where(marked.shift(1).gt(0))
    r = pd.Series(np.nan, index=p.index, name="raw_return")
    held_price = r.copy(); prior_price = r.copy()
    # Vectorized selection rather than per-day dataframe lookups.
    lookup = {code:i for i,code in enumerate(p.columns)}
    ix = np.array([lookup.get(x,0) for x in held]); rows = np.arange(len(p))
    r[:] = within.to_numpy()[rows,ix]; r[held.isna()] = np.nan
    held_price[:] = marked.to_numpy()[rows,ix]; prior_price[:] = marked.shift(1).to_numpy()[rows,ix]
    out = pd.DataFrame({"raw_return":r,"held_contract":held,
        "decision_contract":choice,"held_price":held_price,"prior_price":prior_price})
    out["stale_mark"]=np.isnan(p.to_numpy()[rows,ix]) & held.notna()
    # Prevent tiny-price denominators from silently entering an equity-style return panel.
    out["invalid_return"] = ~np.isfinite(r) | r.le(-1)
    out.loc[out.invalid_return,"raw_return"] = np.nan
    return out

def build(config_path="configs/config.yaml"):
    cfg=configuration(config_path); prepare_directories(); af=author_features()
    universe=pd.read_csv(ROOT/"configs/universe.csv")
    frames=[]; audit=[]
    for row in universe.itertuples():
        dest=ROOT/"data/raw/wrds"/row.ticker
        if not (dest/"bars.parquet").exists():
            audit.append({"ticker":row.ticker,"status":"unavailable"});continue
        bars=pd.read_parquet(dest/"bars.parquet"); meta=pd.read_parquet(dest/"contracts.parquet")
        calendar_fills=0
        if (dest/"calendar_reference.parquet").exists():
            reference=pd.read_parquet(dest/"calendar_reference.parquet")
            reference=reference.dropna(subset=["lasttrddate"]).drop_duplicates("contrdate").set_index("contrdate")
            for column in ["lasttrddate","firstnoticedate"]:
                replacement=meta.contrdate.map(reference[column])
                fill=meta[column].isna()&replacement.notna()
                if column=="lasttrddate":calendar_fills=int(fill.sum())
                meta[column]=meta[column].where(~fill,replacement)
        rolled=rolled_returns(bars,meta,cfg["roll_business_days"])
        rolled.to_parquet(ROOT/f"data/interim/{row.ticker}_rolls.parquet")
        valid=rolled.raw_return.notna()
        r=rolled.loc[valid,"raw_return"]
        if len(r)<400:
            audit.append({"ticker":row.ticker,"status":"insufficient","return_days":len(r)});continue
        # This index is derived from real same-contract P&L; it is not simulated data.
        close=100*(1+r).cumprod()
        f=af(pd.DataFrame({"close":close}, index=close.index))
        f["ticker"]=row.ticker;f["asset_class"]=row.asset_class
        f["raw_return"]=r.reindex(f.index)
        f["raw_target_return"]=r.shift(-1).reindex(f.index)
        f["label_date"]=pd.Series(close.index,index=close.index).shift(-1).reindex(f.index)
        f["published_first_test"]=row.published_first_test
        f["held_contract"]=rolled.held_contract.reindex(f.index)
        f["decision_contract"]=rolled.decision_contract.reindex(f.index)
        # Do not train or score a label that crosses an unknown-return gap.
        next_observation=pd.Series(rolled.index,index=rolled.index).shift(-1)
        f=f[f.label_date.eq(next_observation.reindex(f.index))]
        f=f.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURE_COLUMNS+["target_returns","label_date"])
        frames.append(f)
        audit.append({"ticker":row.ticker,"status":"available","raw_rows":len(bars),
            "return_days":len(r),"feature_days":len(f),"first":str(r.index.min().date()),
            "last":str(r.index.max().date()),"rolls":int(rolled.held_contract.dropna().diff().ne(0).sum()),
            "unknown_return_days":int((~valid).sum()),"stale_mark_days":int(rolled.stale_mark.sum()),
            "calendar_reference_fills":calendar_fills,
            "winsorized_price_days":int((f.srs-f.close).abs().gt(1e-10).sum()),
            "equivalence":row.equivalence})
        print("Features",row.ticker,len(f),flush=True)
    audit_held_prices()
    panel=pd.concat(frames).rename_axis("observation_date").sort_values(["date","ticker"])
    panel.to_parquet(ROOT/"data/processed/features.parquet")
    pd.DataFrame(audit).to_csv(ROOT/"results/replication/tables/data_coverage.csv",index=False)
    print("Prepared",len(panel),"rows",panel.ticker.nunique(),"instruments",flush=True)
    return panel
