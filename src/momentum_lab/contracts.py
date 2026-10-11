"""Build same-contract returns with an explicitly lagged, calendar-based roll rule."""
import numpy as np
import pandas as pd


def continuous_contract_returns(bars, metadata, roll_days=5):
    """Return at t uses the maturity selected at t-1, and OI known at t-2.

    Vendor expiry metadata is assumed known before trading. Exclude inferred last-bar
    expiries; they use hindsight. Never choose the next contract based on its future bar.
    """
    if bars.duplicated(['date','futcode']).any():
        raise ValueError('Duplicate (date, contract) rows; audit instead of keeping an arbitrary value')
    m=metadata.drop_duplicates('futcode').copy()
    if 'ltd_source' in m: m=m[m.ltd_source.eq('vendor')]
    # Only vendor notice/trading dates; course rule dates are approximate and flagged.
    last=pd.to_datetime(m.lasttrddate)
    notice=pd.to_datetime(m.firstnoticedate)
    if 'fnd_is_rule' in m:
        notice=notice.mask(m.fnd_is_rule.fillna(False))
    safe=pd.concat([last,notice],axis=1).min(axis=1)
    safe.index=m.futcode.astype(int)
    p=bars.assign(date=pd.to_datetime(bars.date),futcode=bars.futcode.astype(int))
    price=p.pivot(index='date',columns='futcode',values='settlement').sort_index()
    oi=p.pivot(index='date',columns='futcode',values='oi').reindex_like(price)
    columns=[x for x in price if x in safe.index]
    price=price[columns].astype(float);oi=oi[columns].astype(float)
    if not columns: raise ValueError('No contracts with usable vendor expiry metadata')
    days=pd.Series(price.index,index=price.index)
    eligibility=price.notna() & oi.shift(1).gt(0)
    for code in columns:
        eligibility[code] &= days.add(pd.Timedelta(days=roll_days)).lt(safe[code])
    ranked=sorted(columns,key=lambda code:(safe[code],code))
    # Choose first eligible maturity; preserve no-position days rather than future-fill.
    choice=eligibility[ranked].idxmax(axis=1).where(eligibility.any(axis=1))
    held=choice.shift(1)
    within=price.pct_change(fill_method=None)
    within=within.where(price.gt(0)&price.shift(1).gt(0))
    values=[]
    for t,code in held.items():
        values.append(np.nan if pd.isna(code) else within.at[t,int(code)])
    r=pd.Series(values,index=price.index,name='return',dtype=float)
    return r,held.rename('futcode')
