"""Trailing features; missing warm-up observations stay missing (no backfill)."""
import numpy as np
import pandas as pd
from .data import validate_returns


def volatility(returns, span=60):
    return returns.ewm(span=span, min_periods=span, adjust=True).std()


def tsmom(returns, lookback=252):
    r = validate_returns(returns)
    # Sum log gross returns instead of taking returns of a rolled price splice.
    return np.sign(np.log1p(r).rolling(lookback, min_periods=lookback).sum())


def model_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Author-compatible columns on a synthetic return index; a documented data variant.

    Inputs end at date t. target_returns uses t+1 and must NEVER enter X.
    The label date is retained to purge train/validation/test boundaries.
    """
    frames = []
    for ticker in returns:
        r = returns[ticker]
        # A return index preserves within-contract returns, unlike spliced prices.
        s = (1 + r).cumprod(skipna=False)
        vol = volatility(r)
        f = pd.DataFrame({"ticker": ticker, "srs": s, "daily_returns": r,
                          "daily_vol": vol, "target_returns": .15*r.shift(-1)/(vol*np.sqrt(252))})
        for horizon, name in [(1,"daily"),(21,"monthly"),(63,"quarterly"),
                              (126,"biannual"),(252,"annual")]:
            f[f"norm_{name}_return"] = (s/s.shift(horizon)-1)/(vol*np.sqrt(horizon))
        for short, long in [(8,24),(16,48),(32,96)]:
            # Equivalent to upstream EWM half-life conversion, without future backfill.
            q = (s.ewm(alpha=1/short).mean()-s.ewm(alpha=1/long).mean())/s.rolling(63).std()
            f[f"macd_{short}_{long}"] = q/q.rolling(252).std()
        f["date"] = f.index
        f["label_date"] = pd.Series(f.index, index=f.index).shift(-1)
        f["year"] = f.index.year
        f["day_of_week"] = f.index.dayofweek
        f["day_of_month"] = f.index.day
        f["week_of_year"] = f.index.isocalendar().week.to_numpy()
        f["month_of_year"] = f.index.month
        frames.append(f.replace([np.inf,-np.inf],np.nan).dropna())
    result = pd.concat(frames)
    result.index.name = "Date"
    return result
