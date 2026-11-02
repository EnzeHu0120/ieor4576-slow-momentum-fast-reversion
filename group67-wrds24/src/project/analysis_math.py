"""Post-training calculations for the WRDS 24-series conditional comparison."""
import numpy as np
import pandas as pd


def metrics(returns):
    r=np.asarray(returns,dtype=float)
    if len(r)<2 or not np.isfinite(r).all() or np.any(r<=-1):
        raise ValueError('Invalid returns for performance statistics.')
    wealth=np.cumprod(1+r)
    annual=float(wealth[-1]**(252/len(r))-1)
    vol=float(np.std(r,ddof=1)*np.sqrt(252))
    downside=float(np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252))
    peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
    mdd=float(-np.min(wealth/peak-1))
    positive=r[r>0];negative=r[r<0]
    return {'common_intervals':len(r),'annual_return':annual,'annual_volatility':vol,
            'sharpe_ratio':float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(252)) if vol>0 else np.nan,
            'downside_risk':downside,'sortino_ratio':float(np.mean(r)*252/downside) if downside>0 else np.nan,
            'max_drawdown':mdd,'calmar_ratio':annual/mdd if mdd>0 else np.nan,
            'perc_pos_return':float(np.mean(r>0)),
            'profit_loss_ratio':float(positive.mean()/np.abs(negative).mean()) if len(positive) and len(negative) else np.nan}


def summarize(daily):
    return pd.DataFrame([{'strategy':col,**metrics(daily[col])} for col in daily])


def rescale_ex_post(daily,target=.15):
    """Constant full-sample risk standardization for presentation, not trading."""
    vol=daily.std(ddof=1)*np.sqrt(252)
    if not np.isfinite(vol).all() or not vol.gt(0).all():
        raise ValueError('Cannot standardize a constant or invalid return series.')
    factors=target/vol
    scaled=daily.mul(factors,axis='columns')
    np.testing.assert_allclose(scaled.std(ddof=1)*np.sqrt(252),target,rtol=1e-12,atol=1e-12)
    return scaled,factors


def add_positions(frame):
    x=frame.copy()
    x['long_only_position']=1.
    x['tsmom_annual_position']=np.sign(x.norm_annual_return)
    x['tsmom_monthly_position']=np.sign(x.norm_monthly_return)
    x['tsmom_blend_position']=.5*x.tsmom_annual_position+.5*x.tsmom_monthly_position
    return x


def net_on_common_intervals(full,common_keys,names,cost_bps=(0,1,2,3,4,5)):
    """Compute turnover BEFORE filtering onto common intervals.

    Positions are indexed by their realized target date. Volatility belongs to
    the preceding signal date. Use each asset's immediately previous available
    observation, including observations omitted from the common comparison.
    As in the author's calc_net_returns, first available entry cost is zero.
    Continuous-index futures roll trades cannot be reconstructed here.
    """
    keys=['segment','target_date']
    x=full.sort_values(keys).copy()
    if x.duplicated(keys).any() or not x.daily_vol.gt(0).all():
        raise ValueError('Duplicate dates or invalid volatility in turnover input.')
    common_keys=common_keys[keys].copy()
    joined=common_keys.merge(x,on=keys,how='left',validate='one_to_one')
    if joined.target_returns.isna().any(): raise ValueError('A common interval is missing.')
    raw={};turnover={};tables=[];daily_net={}
    for name in names:
        pos=name+'_position'
        if not np.isfinite(x[pos]).all():raise ValueError('Non-finite position: '+name)
        exposure=x[pos].astype(float)*.15/(x.daily_vol*np.sqrt(252))
        delta=exposure.groupby(x.segment).diff().abs().fillna(0.)
        temporary=x[keys].assign(turnover=delta,gross=x[pos].astype(float)*x.target_returns)
        selected=common_keys.merge(temporary,on=keys,how='left',validate='one_to_one')
        dates=selected.groupby('target_date')
        if not dates.size().eq(24).all():raise ValueError('The common panel is not balanced.')
        raw[name]=dates.gross.sum()/24
        turnover[name]=dates.turnover.sum()/24
        for bps in cost_bps:
            net=raw[name]-bps*1e-4*turnover[name]
            daily_net[f'{name}_{bps}bps']=net
            tables.append({'strategy':name,'cost_bps':bps,
                           'mean_daily_one_way_turnover':float(turnover[name].mean()),**metrics(net)})
    return pd.DataFrame(raw),pd.DataFrame(turnover),pd.DataFrame(daily_net),pd.DataFrame(tables)


def plot_results(raw,scaled,cost_table,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    reference=['long_only','lstm','lstm_cpd_validated']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
    for data,axis,title in [(raw,axes[0],'Gross conditional comparison'),
                            (scaled,axes[1],'Ex-post standardization to 15% volatility')]:
        for name in reference:
            if name in data:axis.plot(data.index,(1+data[name]).cumprod(),label=name,lw=1.5)
        axis.set(title=title,ylabel='Growth of 1 (474 retained intervals)')
        axis.grid(alpha=.2);axis.legend(fontsize=8);axis.tick_params(axis='x',rotation=25)
    fig.savefig(output/'cumulative_comparison.png',dpi=180);plt.close(fig)
    fig,axis=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for name,g in cost_table.groupby('strategy',sort=False):
        if name in ['tsmom_blend','tsmom_monthly']:continue
        axis.plot(g.cost_bps,g.sharpe_ratio,marker='o',ms=3,label=name)
    axis.axhline(0,color='black',lw=.7)
    axis.set(xlabel='One-way transaction cost (basis points)',ylabel='Conditional annualized Sharpe',
             title='Turnover cost sensitivity on retained common intervals')
    axis.legend(fontsize=7,ncol=2);axis.grid(alpha=.2)
    fig.savefig(output/'cost_sensitivity.png',dpi=180);plt.close(fig)
