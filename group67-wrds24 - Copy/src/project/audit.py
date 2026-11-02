"""Independent read-only audit of the uploaded post-training result archive."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit
from scipy.optimize import brentq

from .paths import ROOT
R=ROOT/'data/interim/artifacts/supplement'; C=ROOT/'data/interim/artifacts/cpd'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def maximum_error(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def performance(r):
    r=np.asarray(r,dtype=float);wealth=np.cumprod(1+r);sd=r.std(ddof=1)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252)
    dd=-np.min(wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1)
    annual=wealth[-1]**(252/len(r))-1
    return dict(common_intervals=len(r),annual_return=annual,annual_volatility=sd*np.sqrt(252),
                sharpe_ratio=r.mean()/sd*np.sqrt(252),downside_risk=downside,
                sortino_ratio=r.mean()*252/downside,max_drawdown=dd,calmar_ratio=annual/dd,
                perc_pos_return=np.mean(r>0),profit_loss_ratio=r[r>0].mean()/np.abs(r[r<0]).mean())

complete=load(R/'audit_inputs/CPD_COMPLETE.json'); cfg=load(C/'run_config.json')
assert sha(R/'audit_inputs/features_cpd.csv')==cfg['feature_sha256']==complete['cpd_features_sha256']
assert sha(R/'audit_inputs/features.csv')==complete['config']['features_sha256']==load(C/'frozen_lstm/run_config.json')['feature_sha256']
assert sha(R/'audit_inputs/segmented_levels.csv')==complete['config']['levels_sha256']
assert sha(R/'audit_inputs/changepoint_detection.py')==complete['config']['author_cpd_sha256']
assert sha(R/'audit_inputs/run_wrds_panel24_analysis.py')==sha(ROOT/'src/project/author_code/run_wrds_panel24_analysis.py')
assert sha(R/'audit_inputs/paper67_panel24_analysis.py')==sha(ROOT/'src/project/author_code/paper67_panel24_analysis.py')
f=pd.read_csv(R/'audit_inputs/features_cpd.csv',parse_dates=['signal_date','target_date'])
base=pd.read_csv(R/'audit_inputs/features.csv',parse_dates=['signal_date','target_date'])
levels=pd.read_csv(R/'audit_inputs/segmented_levels.csv',parse_dates=['signal_date'])
diag=pd.read_csv(R/'audit_inputs/fit_windows.csv',parse_dates=['date'])
assert len(f)==55839 and len(diag)==279195
assert len(diag)==len(f)*5 and not diag.duplicated(['segment','date','lookback']).any()
base_roundtrip_error=0.
for column in base:
    if pd.api.types.is_numeric_dtype(base[column]) and not pd.api.types.is_bool_dtype(base[column]):
        base_roundtrip_error=max(base_roundtrip_error,maximum_error(f[column],base[column]))
        np.testing.assert_allclose(f[column],base[column],rtol=1e-12,atol=1e-12)
    else: assert f[column].equals(base[column]),column
assert (f.signal_date<f.target_date).all() and f.target_date.lt('2021-01-01').all()
indexed={s:g.sort_values('position').reset_index(drop=True) for s,g in levels.groupby('segment')}
causal_fills=[]
for lb in [10,21,63,126,252]:
    j=f.merge(diag[diag.lookback.eq(lb)],left_on=['segment','signal_date'],right_on=['segment','date'],validate='one_to_one',how='left').sort_values(['segment','source_position']).reset_index(drop=True)
    assert j.status.notna().all(); ok=j.status.eq('completed')
    assert j.loc[ok,f'cp_score_{lb}'].between(-1e-8,1+1e-8).all()
    assert j.loc[ok,f'cp_rl_{lb}'].between(-1e-8,1+1e-8).all()
    expected=j.signal_date.where(ok).groupby(j.segment).ffill()
    assert pd.to_datetime(j[f'cp_source_date_{lb}']).eq(expected).all()
    expected_score=j[f'cp_score_{lb}'].where(ok).groupby(j.segment).ffill()
    anchor=(j.source_position-lb*j[f'cp_rl_{lb}']).where(ok).groupby(j.segment).ffill()
    np.testing.assert_allclose(j[f'cp_score_{lb}'],expected_score,atol=1e-12,rtol=0)
    np.testing.assert_allclose(j[f'cp_rl_{lb}'],(j.source_position-anchor)/lb,atol=1e-10,rtol=1e-10)
    for row in j[~ok].itertuples():
        causal_fills.append((row.segment,lb,row.signal_date.strftime('%Y-%m-%d'),getattr(row,f'cp_source_date_{lb}')))
manifest_fills={(x['segment'],x['lookback'],x['date'],x['source_date']) for x in complete['failed_fits_filled_from_past']}
assert set(causal_fills)==manifest_fills and len(causal_fills)==154

# Reconstruct marginal likelihoods from sampled fitted parameters and the
# actual raw-index windows. No optimizer, TensorFlow or GPflow is required.
def matern(x,variance,length):
    distance=np.abs(x[:,None]-x[None,:])*np.sqrt(3)/float(length)
    return float(variance)*(1+distance)*np.exp(-distance)
def nlml(k,noise,y):
    cov=k+np.eye(len(y))*float(noise)
    factor=cho_factor(cov,lower=True,check_finite=True)
    return .5*float(y@cho_solve(factor,y))+np.log(np.diag(factor[0])).sum()+.5*len(y)*np.log(2*np.pi)
score_errors=[]
for p in sorted((R/'audit_inputs/cache_samples').rglob('*.json')):
    a=load(p);g=indexed[a['segment']];t=a['t'];lb=a['lookback']
    assert a['observations']==lb+1
    assert g.loc[t,'signal_date'].strftime('%Y-%m-%d')==a['date']==a['window_end_date']
    assert g.loc[t-lb,'signal_date'].strftime('%Y-%m-%d')==a['window_start_date']
    if a['status']!='completed':continue
    raw=g.close.pct_change(fill_method=None).iloc[t-lb:t+1].to_numpy()
    y=(raw-raw.mean())/raw.std(ddof=0);x=np.arange(t-lb,t+1,dtype=float)
    m=a['matern_parameters'];c=a['changepoint_parameters']
    km=matern(x,m['kM_variance'],m['kM_lengthscales'])
    sig=expit(float(c['kC_steepness'])*(x-float(c['kC_changepoint_location'])))
    kc=matern(x,c['k1_variance'],c['k1_lengthscale'])*np.outer(1-sig,1-sig)+matern(x,c['k2_variance'],c['k2_lengthscale'])*np.outer(sig,sig)
    expected_score=expit(nlml(km,m['kM_likelihood_variance'],y)-nlml(kc,c['kC_likelihood_variance'],y))
    score_errors.append(abs(float(expected_score)-a['cp_score']))
    assert abs((t-a['cp_location'])/lb-a['cp_location_norm'])<1e-8
assert len(score_errors)>450 and max(score_errors)<1e-6,max(score_errors)

allp=pd.read_csv(R/'all_available_test_asset_predictions.csv',parse_dates=['signal_date','target_date'])
common=pd.read_csv(C/'asset_comparison_2019_2020.csv',parse_dates=['signal_date','target_date'])
assert len(allp)==12162 and not allp.duplicated(['segment','target_date']).any()
assert allp.code.nunique()==24 and (allp.signal_date<allp.target_date).all()
assert allp.target_date.between('2019-01-01','2020-12-31').all()
keys=['segment','target_date'];retained=common[keys].merge(allp,on=keys,validate='one_to_one')
assert len(retained)==11376
names=['long_only','tsmom_annual','tsmom_blend','tsmom_monthly','macd','lstm']+[f'lstm_cpd_{l}' for l in [10,21,63,126,252]]+['lstm_cpd_validated']
for name in names:assert np.isfinite(allp[name+'_position']).all() and allp[name+'_position'].abs().le(1+1e-6).all()
np.testing.assert_array_equal(allp.lstm_cpd_252_position,allp.lstm_cpd_validated_position)
g=pd.read_csv(R/'gross_common_interval_returns.csv',index_col=0,parse_dates=True)
t=pd.read_csv(R/'one_way_turnover_common_intervals.csv',index_col=0,parse_dates=True)
net=pd.read_csv(R/'cost_adjusted_common_interval_returns.csv',index_col=0,parse_dates=True)
cost=pd.read_csv(R/'cost_sensitivity_0_to_5bps.csv')
gm=pd.read_csv(R/'gross_fixed_lookbacks.csv').set_index('strategy')
sm=pd.read_csv(R/'ex_post_15pct_metrics.csv').set_index('strategy')
scaled=pd.read_csv(R/'ex_post_15pct_returns.csv',index_col=0,parse_dates=True)
factors=pd.read_csv(R/'ex_post_15pct_multipliers.csv',index_col=0).iloc[:,0]
metric_error=turnover_error=net_error=0.
for name in names:
    gross=(retained[name+'_position']*retained.target_returns).groupby(retained.target_date).sum()/24
    np.testing.assert_allclose(gross,g[name],atol=1e-12,rtol=0)
    x=allp.sort_values(['segment','target_date']).copy()
    exposure=x[name+'_position']*.15/(x.daily_vol*np.sqrt(252))
    x['turnover']=exposure.groupby(x.segment).diff().abs().fillna(0.)
    selected=common[keys].merge(x[keys+['turnover']],on=keys,validate='one_to_one')
    turnover=selected.groupby('target_date').turnover.sum()/24
    turnover_error=max(turnover_error,maximum_error(turnover,t[name]));assert turnover_error<1e-12
    scale=.15/(g[name].std(ddof=1)*np.sqrt(252))
    assert abs(scale-factors[name])<1e-12
    np.testing.assert_allclose(scaled[name],g[name]*scale,atol=1e-12,rtol=0)
    for data,table in [(g[name],gm),(scaled[name],sm)]:
        for key,value in performance(data).items():
            metric_error=max(metric_error,abs(float(table.loc[name,key])-float(value)))
    for bps in range(6):
        expected=g[name]-turnover*bps*1e-4
        net_error=max(net_error,maximum_error(expected,net[f'{name}_{bps}bps']))
        row=cost[cost.strategy.eq(name)&cost.cost_bps.eq(bps)].iloc[0]
        for key,value in performance(expected).items():metric_error=max(metric_error,abs(float(row[key])-float(value)))
assert metric_error<1e-10 and net_error<1e-12
assert maximum_error(sm.annual_volatility,.15)<1e-12
assert maximum_error(sm.sharpe_ratio,gm.sharpe_ratio)<1e-12
replays={}
for original,name in [('lstm','lstm'),('lstm_cpd','lstm_cpd_validated')]:
    merged=common[keys+[original+'_position']].merge(allp[keys+[name+'_position']],on=keys,validate='one_to_one',suffixes=('_old','_new'))
    a,b=(original+'_position_old',name+'_position_new') if original==name else (original+'_position',name+'_position')
    replays[original]=maximum_error(merged[a],merged[b]);assert replays[original]<1e-7
ckpt=pd.read_csv(R/'validation_selected_checkpoints.csv')
for row in ckpt.itertuples():
    if row.strategy=='lstm':weight=C/'frozen_lstm/best_lstm.weights.h5'
    else:weight=C/f'cpd_{int(row.lookback)}'/f'trial_{row.trial:03d}'/'best.weights.h5'
    assert sha(weight)==row.checkpoint_sha256

def sharpe(name,bps):
    r=g[name]-bps*1e-4*t[name];return float(r.mean()/r.std(ddof=1)*np.sqrt(252))
crossover=brentq(lambda b:sharpe('lstm_cpd_validated',b)-sharpe('long_only',b),0,1)
zero=brentq(lambda b:sharpe('lstm_cpd_validated',b),4,5)
report={'status':'PASSED','source_archive_sha256':sha(ROOT/'data/raw/artifacts/paper67_panel24_supplement_results.zip'),
 'gp_windows':len(diag),'gp_features':len(f),'causal_failed_fit_fills':len(causal_fills),
 'failed_fit_fraction':len(causal_fills)/len(diag),'sample_numeric_gp_likelihood_reconstructions':len(score_errors),
 'sample_gp_score_reconstruction_max_error':max(score_errors),'fitted_parameters_refitted':False,
 'all_available_test_prediction_rows':len(allp),'retained_common_asset_rows':len(retained),'common_intervals':len(g),
 'position_replay_max_errors':replays,'all_checkpoint_hashes_match_original_run':True,
 'turnover_recomputation_max_error':turnover_error,'net_return_recomputation_max_error':net_error,
 'all_gross_scaled_and_cost_metric_recomputation_max_error':metric_error,
 'cpd_vs_long_only_sharpe_cost_crossover_bps':crossover,'cpd_zero_sharpe_cost_bps':zero,
 'main_findings':['Validation selected 252-day CPD, not selected using the test results.',
                  'Gross CPD Sharpe is 1.880281 versus ordinary LSTM -0.348419 and long-only 1.682256.',
                  'At 1 bps one-way cost CPD Sharpe falls to 1.499530, below long-only 1.663770.',
                  'CPD 21-day OOS Sharpe is -0.509874; CPD is not uniformly helpful across lookbacks.',
                  'All 15% ex-post standardized series preserve Sharpe; this is presentation scaling, not causal risk targeting.'],
 'limits':['Only sampled GP likelihoods were reconstructed; all source dates and 154 past-only fills were checked.',
           'No new training or independent GPU hardware inference was performed here; uploaded GPU replay outputs agree with earlier saved outputs.',
           'All performance and cost results remain conditional on 474 common intervals, not full-calendar executable returns.',
           'Single fixed split, seed, and WRDS 24-series universe; no claim of statistical or universal CPD superiority.']}
(ROOT/'results/replication/tables/independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
