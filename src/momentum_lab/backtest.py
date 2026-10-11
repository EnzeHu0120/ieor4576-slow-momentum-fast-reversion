"""Exposure simulation, kept separate from fitting and signal construction."""
import numpy as np
from .costs import traded_notional, net_returns
from .features import volatility
from .data import validate_returns


def run_backtest(returns, signals, bps=1., contracts=None, execution_lag=2,
                 target_vol=.15, max_asset_exposure=2.):
    r = validate_returns(returns)
    if not signals.index.equals(r.index) or not signals.columns.equals(r.columns):
        raise ValueError("Signal dates/assets must match returns")
    if execution_lag < 1 or target_vol <= 0 or max_asset_exposure <= 0:
        raise ValueError("Require lag>=1 and positive risk limits")
    if (signals.abs() > 1).any().any():
        raise ValueError("Position signals must be in [-1,1]")
    n = len(r.columns)  # Fixed denominator; unavailable assets leave capital in cash.
    target = target_vol*signals/(np.sqrt(252)*volatility(r)*n)
    target = target.clip(-max_asset_exposure/n,max_asset_exposure/n).fillna(0.)
    # lag=2: observe close t, transact at close t+1, earn close t+1 -> t+2.
    # lag=1 is an idealised paper alignment, not an achievable settlement fill.
    weights = target.shift(execution_lag).fillna(0.)
    if (weights.ne(0) & r.isna()).any().any():
        raise ValueError("Missing return while holding exposure; audit data before backtesting")
    gross = (weights*r.fillna(0.)).sum(axis=1)
    trades = traded_notional(weights, contracts)
    net = net_returns(gross, trades, bps)
    if (net <= -1).any():
        raise ValueError("Equity exhausted: inspect leverage and contract-dollar P&L")
    return {"gross":gross, "net":net, "weights":weights, "trades":trades,
            "turnover":trades.sum(axis=1)}
