"""Costs are one-way traded notional; flips and futures rolls pay for both legs."""
import numpy as np
import pandas as pd


def traded_notional(weights: pd.DataFrame, contracts: pd.DataFrame | None = None):
    if weights.isna().any().any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Weights must be finite")
    old = weights.shift(1).fillna(0.0)
    trades = (weights-old).abs()
    if contracts is not None:
        if not contracts.index.equals(weights.index) or not contracts.columns.equals(weights.columns):
            raise ValueError("Contract IDs must align exactly with effective holdings")
        if (weights.ne(0) & contracts.isna()).any().any():
            raise ValueError("Nonzero holding without a contract ID")
        # Netting exposures across different maturities hides the exit and entry fees.
        rolled = contracts.ne(contracts.shift(1)) & old.ne(0) & weights.ne(0)
        trades = trades.mask(rolled, old.abs()+weights.abs())
    return trades


def net_returns(gross, trades, bps):
    if bps < 0 or not np.isfinite(bps):
        raise ValueError("Cost in bps must be finite and nonnegative")
    return gross - trades.sum(axis=1)*bps*1e-4


def contract_fee(delta_contracts, commission_per_side, exchange_per_side,
                 spread_ticks, slippage_ticks, tick_value):
    """Dollar estimate from explicit contract specs; no invented broker fee defaults."""
    p = [commission_per_side, exchange_per_side, spread_ticks, slippage_ticks, tick_value]
    if not all(np.isfinite(x) and x >= 0 for x in p):
        raise ValueError("Supply nonnegative fees and tick specifications")
    return abs(delta_contracts)*(commission_per_side+exchange_per_side+
                                (spread_ticks/2+slippage_ticks)*tick_value)
