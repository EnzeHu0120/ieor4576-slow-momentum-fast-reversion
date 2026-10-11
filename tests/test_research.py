import numpy as np
import pandas as pd
import pytest
from momentum_lab.data import synthetic_returns,validate_returns
from momentum_lab.features import tsmom,model_features
from momentum_lab.backtest import run_backtest
from momentum_lab.costs import traded_notional,net_returns,contract_fee
from momentum_lab.evaluation import metrics
from momentum_lab.splits import Fold,masks


def test_fee_flip_initial_and_roll():
    idx=pd.date_range('2020-01-01',periods=4)
    w=pd.DataFrame({'A':[1.,1.,-1.,0.]},index=idx)
    ids=pd.DataFrame({'A':['OLD','NEW','NEW','NEW']},index=idx)
    trade=traded_notional(w,ids)
    np.testing.assert_allclose(trade.A,[1,2,2,1])
    np.testing.assert_allclose(net_returns(pd.Series(0.,index=idx),trade,1),[-.0001,-.0002,-.0002,-.0001])
    assert contract_fee(-2,1,.5,2,1,10)==43


def test_future_perturbation_cannot_change_past_positions():
    r=synthetic_returns(n=900);changed=r.copy();changed.iloc[700:]*=-1
    a=run_backtest(r,tsmom(r))['weights'];b=run_backtest(changed,tsmom(changed))['weights']
    pd.testing.assert_frame_equal(a.iloc[:702],b.iloc[:702])
    fa=model_features(r);fb=model_features(changed)
    cols=[x for x in fa if x.startswith(('norm_','macd_'))]+['daily_vol']
    pd.testing.assert_frame_equal(fa.loc[fa.date<r.index[700],cols],fb.loc[fb.date<r.index[700],cols])


def test_close_timing_and_missing_held_return():
    r=synthetic_returns(n=200,assets=1);s=r*0+1
    bt=run_backtest(r,s,bps=0,execution_lag=2)
    expected=.15/(r.ewm(span=60,min_periods=60).std()*np.sqrt(252))
    expected=expected.clip(upper=2).shift(2).fillna(0.)
    pd.testing.assert_frame_equal(bt['weights'],expected)
    np.testing.assert_allclose(bt['gross'],(expected*r).sum(axis=1))
    r.iloc[-1,0]=np.nan
    with pytest.raises(ValueError,match='Missing return'):run_backtest(r,s)


def test_boundary_labels_and_duplicate_dates():
    f=pd.DataFrame({'date':pd.to_datetime(['2014-12-31','2015-01-01']),
                    'label_date':pd.to_datetime(['2015-01-01','2015-01-02'])})
    train,val,test=masks(f,Fold('2010-01-01','2014-01-01','2015-01-01','2020-01-01'))
    assert not val.iloc[0] and test.iloc[1]
    r=synthetic_returns(n=10);r.index=[r.index[0]]*10
    with pytest.raises(ValueError):validate_returns(r)


def test_initial_drawdown_and_cost_monotonicity():
    idx=pd.date_range('2020-01-01',periods=2)
    assert metrics(pd.Series([-.1,0],index=idx))['max_drawdown']==pytest.approx(-.1)
    r=synthetic_returns(n=900);s=tsmom(r)
    a=run_backtest(r,s,bps=0)['net'];b=run_backtest(r,s,bps=5)['net']
    assert (b<=a).all()


def test_roll_return_never_crosses_contract_price_levels():
    from momentum_lab.contracts import continuous_contract_returns
    dates=pd.date_range('2020-01-01',periods=10)
    bars=pd.concat([pd.DataFrame({'date':dates,'futcode':c,'settlement':p,'oi':10.})
                    for c,p in [(1,100.),(2,200.)]])
    meta=pd.DataFrame({'futcode':[1,2],'lasttrddate':pd.to_datetime(['2020-01-10','2020-02-01']),
                       'firstnoticedate':[pd.NaT,pd.NaT]})
    r,held=continuous_contract_returns(bars,meta,roll_days=5)
    assert held.dropna().nunique()==2
    assert r.dropna().eq(0).all()
    changed=bars.copy();changed.loc[changed.date>=dates[7],'oi']=0
    _,held2=continuous_contract_returns(changed,meta,roll_days=5)
    pd.testing.assert_series_equal(held.iloc[:8],held2.iloc[:8])
