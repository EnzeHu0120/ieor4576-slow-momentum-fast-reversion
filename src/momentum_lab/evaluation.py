"""Daily excess-P&L metrics; cash interest and margin are not included."""
import numpy as np


def metrics(r, turnover=None):
    r = r.dropna()
    if len(r)<2 or (r<=-1).any():
        raise ValueError("Need two valid returns above -100%")
    wealth = (1+r).cumprod()
    # Include initial wealth=1 so a loss on the first day counts as drawdown.
    peak = wealth.cummax().clip(lower=1.)
    sd = r.std(ddof=1)
    return {"days":len(r), "start":str(r.index.min().date()), "end":str(r.index.max().date()),
            "annual_return":float(wealth.iloc[-1]**(252/len(r))-1),
            "annual_vol":float(sd*np.sqrt(252)),
            "sharpe":float(r.mean()/sd*np.sqrt(252)) if sd>0 else None,
            "max_drawdown":float((wealth/peak-1).min()),
            "annual_one_way_turnover":float(turnover.reindex(r.index).mean()*252)
                if turnover is not None else None}
