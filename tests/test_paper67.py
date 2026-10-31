"""Only checks that protect the research conclusions."""
import numpy as np
import pandas as pd
from momentum_lab.cpd import dense_value_gradient,fast_likelihood
from momentum_lab.dataset import rolled_returns
from momentum_lab.models import sharpe_loss,sequence_data
from momentum_lab.study_evaluation import aggregate,performance
import torch
import pytest

def test_gp_state_space_equals_dense_changepoint_covariance():
    # Fixed observations for a numerical identity, not an experimental dataset.
    y=np.array([-.4,.2,.7,-.3,.1,.5,-1.,.8,.2,-.1,.6,-.5])
    theta=np.array([.1,.3,-.4,.7,.6,4.2,-.2])
    assert abs(fast_likelihood(theta,y,True)-dense_value_gradient(theta,y,True)[0])<1e-8

def test_roll_does_not_book_near_far_price_difference():
    dates=pd.bdate_range("2020-01-01",periods=15)
    bars=pd.concat([pd.DataFrame({"date":dates,"futcode":code,"settlement":price,"oi":100.,"volume":10.})
        for code,price in [(1,100.),(2,200.)]],ignore_index=True)
    metadata=pd.DataFrame({"futcode":[1,2],"lasttrddate":pd.to_datetime(["2020-01-15","2020-03-01"]),
        "firstnoticedate":[pd.NaT,pd.NaT]})
    r=rolled_returns(bars,metadata)
    assert r.held_contract.nunique()==2
    assert r.raw_return.dropna().eq(0).all()

def test_sharpe_loss_matches_paper_pooled_asset_time_objective():
    p=torch.tensor([[.2,-.4],[.8,.5]],requires_grad=True)
    y=torch.tensor([[.01,-.02],[.03,-.01]])
    realized=(p.detach().numpy()*y.numpy()).ravel()
    expected=-np.sqrt(252)*realized.mean()/np.sqrt(np.mean(realized**2)-realized.mean()**2+1e-9)
    loss=sharpe_loss(p,y);assert abs(loss.item()-expected)<1e-5
    loss.backward();assert torch.isfinite(p.grad).all()

def test_training_and_validation_labels_are_purged_at_boundaries():
    dates=pd.bdate_range('2010-01-01',periods=1100)
    f=pd.DataFrame({'ticker':'A','date':dates,'label_date':dates+pd.offsets.BDay(1),
        'published_first_test':2015,'target_returns':.001,'signal':.1})
    _,eligible,splits=sequence_data(f,2015,{'sequence_length':63,'train_fraction':.9},['signal'])
    assert eligible==['A']
    row=splits[0]
    assert pd.Timestamp(row['training_last_label'])<pd.Timestamp(row['validation_first'])
    assert pd.Timestamp(row['validation_last_label'])<pd.Timestamp(row['test_start'])

def test_reversal_and_both_roll_legs_pay_actual_turnover(tmp_path,monkeypatch):
    import momentum_lab.study_evaluation as evaluation
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    (tmp_path/'results/replication').mkdir(parents=True)
    p=pd.DataFrame({'strategy':'LSTM','ticker':'A','fold':2020,
        'label_date':pd.bdate_range('2020-01-01',periods=4),'position':[1.,1.,-1.,0.],
        'daily_vol':.15/np.sqrt(252),'target_returns':0.,'raw_target_return':0.,
        'decision_contract':['OLD','NEW','NEW','NEW']})
    d=aggregate(p,{'target_volatility':.15})
    np.testing.assert_allclose(d.paper_turnover,[1,0,2,1])
    np.testing.assert_allclose(d.roll_turnover,[1,2,2,1])
    assert (d.gross-2.5e-4*d.roll_turnover<=d.gross).all()

def test_drawdown_includes_initial_capital_loss():
    assert performance([-.1,0.])['max_drawdown']==pytest.approx(.1)
