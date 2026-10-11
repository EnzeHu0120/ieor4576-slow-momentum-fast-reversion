"""Explicit input contracts: never forward-fill a missing tradable return."""
from pathlib import Path
import numpy as np
import pandas as pd


def validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.copy().astype(float)
    if not isinstance(r.index, pd.DatetimeIndex) or r.index.has_duplicates:
        raise ValueError("Returns require a unique DatetimeIndex")
    if not r.index.is_monotonic_increasing or r.columns.has_duplicates:
        raise ValueError("Sort dates and use unique asset names")
    if r.empty or np.isinf(r.to_numpy()).any():
        raise ValueError("Empty input or infinite return")
    if (r <= -1).any().any():
        raise ValueError("Return <= -100%: use contract-dollar P&L for nonpositive prices")
    return r


def load_returns(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    r = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0)
    r.index = pd.to_datetime(r.index)
    return validate_returns(r)


def synthetic_returns(seed=67, n=1600, assets=4) -> pd.DataFrame:
    """Deterministic plumbing demo; never presented as market or replication data."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.00015, 0.01, (n, assets))
    return pd.DataFrame(x, index=pd.bdate_range("2010-01-04", periods=n),
                        columns=[f"SYNTH_{i}" for i in range(assets)])
