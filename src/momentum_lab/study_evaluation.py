"""Exhibits 3/4, matched OOS baselines, Appendix C, and roll-aware risk diagnostics."""
import json
import numpy as np
import pandas as pd
from .paths import ROOT, configuration

def performance(r):
    r=pd.Series(r).dropna().astype(float);vol=r.std(ddof=1)*np.sqrt(252)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252)
    wealth=(1+r).cumprod();peak=wealth.cummax().clip(lower=1.)
    dd=float((1-wealth/peak).max())
    annual=float(np.expm1(np.log1p(r).sum()*252/len(r))) if r.gt(-1).all() else np.nan
    return {"annual_return":annual,"annual_volatility":vol,
        "sharpe":r.mean()*252/vol if vol else np.nan,"downside_deviation":downside,
        "sortino":r.mean()*252/downside if downside else np.nan,
        "max_drawdown":dd,"calmar":annual/dd if dd else np.nan,
        "positive_fraction":r.gt(0).mean(),
        "profit_loss_ratio":r[r>0].mean()/(-r[r<0].mean()) if r.lt(0).any() else np.nan,
        "days":len(r)}

def prediction_panel(cfg=None):
    cfg=cfg or configuration()
    files=[];compatible_receipts=[]
    for path in sorted((ROOT/"results/replication/checkpoints").glob("*/*/receipt.json")):
        d=json.loads(path.read_text())
        if d.get("config")==cfg and len(d["trials"])==cfg["random_search_trials"]:
            files.append(path.parent/"predictions.parquet");compatible_receipts.append(d)
    if not files:raise FileNotFoundError("No completed model experiments")
    neural=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    base=neural[neural.strategy.eq("LSTM")].copy()
    features=pd.read_parquet(ROOT/"data/processed/features.parquet").reset_index(drop=True)
    macd=["macd_8_24","macd_16_48","macd_32_96"]
    base=base.merge(features[["ticker","date"]+macd],on=["ticker","date"],validate="one_to_one")
    panels=[neural]
    # The author's calc_combined_signal returns a trend estimate, not a position.
    # Apply its separate Baz position-sizing function (2019 reference, eq. 7).
    macd_trend=base[macd].mean(axis=1)
    for name,position in {"Long Only":np.ones(len(base)),
        "MACD":macd_trend*np.exp(-macd_trend**2/4)/.89,
        "TSMOM w=0":np.sign(base.norm_annual_return),
        "TSMOM w=0.5":.5*(np.sign(base.norm_annual_return)+np.sign(base.norm_monthly_return)),
        "TSMOM w=1":np.sign(base.norm_monthly_return)}.items():
        p=base.drop(columns=macd).copy();p["strategy"]=name;p["position"]=position;panels.append(p)
    p=pd.concat(panels,ignore_index=True)
    # Structural LBW selection uses pre-test validation loss only, not OOS Sharpe.
    receipts=[]
    for d in compatible_receipts:
        if d['strategy'].startswith('LSTM-CPD-'):
            receipts.append({"strategy":d["strategy"],"fold":d["fold"],"validation_loss":d["validation_loss"]})
    if receipts:
        candidates=pd.DataFrame(receipts)
        candidates=candidates[candidates.groupby('fold').strategy.transform('nunique').eq(len(cfg['cpd_windows']))]
        if len(candidates):
            selected=candidates.sort_values("validation_loss").groupby("fold",as_index=False).first()
            selected.to_csv(ROOT/"results/replication/tables/selected_cpd_windows.csv",index=False)
            q=p.merge(selected[["strategy","fold"]],on=["strategy","fold"])
            q["selected_strategy"]=q.strategy;q["strategy"]="LSTM-CPD-Optimised";p=pd.concat([p,q],ignore_index=True)
        else:
            (ROOT/"results/replication/tables/selected_cpd_windows.csv").unlink(missing_ok=True)
    return p.sort_values(["strategy","ticker","label_date"]).reset_index(drop=True)

def aggregate(p,cfg):
    q=p.copy()
    q["N"]=q.groupby(["strategy","fold"]).ticker.transform("nunique")
    q["exposure"]=q.position*cfg["target_volatility"]/(q.daily_vol*np.sqrt(252))
    q["gross"]=q.position*q.target_returns/q.N
    q["tradable_gross"]=q.exposure*q.raw_target_return/q.N
    # Actual portfolio exposure includes 1/N; this charges changing eligibility too.
    q["portfolio_exposure"]=q.exposure/q.N
    group=q.groupby(["strategy","ticker"],sort=False)
    old=group.portfolio_exposure.shift(1).fillna(0.)
    q["paper_turnover"]=(q.portfolio_exposure-old).abs()
    changed=q.decision_contract.ne(group.decision_contract.shift(1)) & old.ne(0)
    q["roll_turnover"]=q.paper_turnover.where(~changed,q.portfolio_exposure.abs()+old.abs())
    q["roll_extra"]=q.roll_turnover-q.paper_turnover
    # Conservative extension: at t use yesterday's signal and resize; compare the
    # same actual held contract's raw return. Signals remain model-trained gross.
    q["delayed_position"]=group.position.shift(1).fillna(0.)
    q["delayed_exposure"]=q.delayed_position*cfg["target_volatility"]/(q.daily_vol*np.sqrt(252))/q.N
    q["delayed_gross"]=q.delayed_exposure*q.raw_target_return
    previous=group.delayed_exposure.shift(1).fillna(0.)
    q["delayed_turnover"]=(q.delayed_exposure-previous).abs().where(~changed,q.delayed_exposure.abs()+previous.abs())
    extension_cfg=configuration("configs/extension.yaml")
    cap=extension_cfg["exposure_cap_per_instrument"]
    q["capped_exposure"]=q.exposure.clip(-cap,cap)/q.N
    q["capped_gross"]=q.capped_exposure*q.raw_target_return
    old_cap=group.capped_exposure.shift(1).fillna(0.)
    q["capped_turnover"]=(q.capped_exposure-old_cap).abs().where(~changed,q.capped_exposure.abs()+old_cap.abs())
    # Close remaining positions at each instrument's last scored observation.
    terminal=~q.duplicated(["strategy","ticker"],keep="last")
    q.loc[terminal,"paper_turnover"]+=q.loc[terminal,"portfolio_exposure"].abs()
    q.loc[terminal,"roll_turnover"]+=q.loc[terminal,"portfolio_exposure"].abs()
    q.loc[terminal,"delayed_turnover"]+=q.loc[terminal,"delayed_exposure"].abs()
    q.loc[terminal,"capped_turnover"]+=q.loc[terminal,"capped_exposure"].abs()
    daily=q.groupby(["strategy","fold","label_date"],as_index=False)[[
        "gross","tradable_gross","paper_turnover","roll_turnover","roll_extra","delayed_gross","delayed_turnover","capped_gross","capped_turnover"]].sum()
    q.to_parquet(ROOT/"results/replication/asset_predictions.parquet",index=False)
    return daily

def evaluate(config_path="configs/config.yaml"):
    cfg=configuration(config_path);p=prediction_panel(cfg);daily=aggregate(p,cfg)
    daily.to_parquet(ROOT/"results/replication/daily_returns.parquet",index=False)
    window_vol=daily[daily.fold.lt(2020)].groupby(['strategy','fold']).gross.std(ddof=1)*np.sqrt(252)
    scaling=(cfg['target_volatility']/window_vol.groupby(level='strategy').mean()).to_dict()
    pd.Series(scaling,name='global_ex_post_factor').rename_axis('strategy').to_csv(ROOT/'results/replication/tables/volatility_rescaling.csv')
    records=[];costs=[];extension=[];years=[]
    for (strategy,fold),g in daily.groupby(["strategy","fold"]):
        r=g.set_index("label_date").gross
        for exhibit,scaled in [(3,r),(4,r*scaling[strategy])]:
            records.append({"strategy":strategy,"fold":fold,"exhibit":exhibit,**performance(scaled)})
        for c in cfg["cost_bps"]:
            costs.append({"strategy":strategy,"fold":fold,"cost_bps":c,
                **performance(g.gross-c*1e-4*g.paper_turnover),
                "annual_turnover":g.paper_turnover.mean()*252})
            for scenario,rcol,tcol in [("raw returns plus rolls","tradable_gross","roll_turnover"),
                ("one-session signal delay plus rolls","delayed_gross","delayed_turnover"),
                ("2x exposure cap plus rolls","capped_gross","capped_turnover")]:
                extension.append({"strategy":strategy,"fold":fold,"cost_bps":c,"scenario":scenario,
                    **performance(g[rcol]-c*1e-4*g[tcol]),
                    "annual_turnover":g[tcol].mean()*252,"annual_roll_extra":g.roll_extra.mean()*252})
        for year,part in g.groupby(g.label_date.dt.year):
            years.append({"strategy":strategy,"year":year,**performance(part.gross)})
    fold=pd.DataFrame(records);fold.to_csv(ROOT/"results/replication/tables/exhibits_by_fold.csv",index=False)
    core=fold[fold.fold<2020]
    for exhibit in [3,4]:
        group=core[core.exhibit.eq(exhibit)].groupby("strategy")
        table=group.mean(numeric_only=True).drop(columns=["fold","exhibit"])
        table["test_windows"]=group.fold.nunique()
        table.to_csv(ROOT/f"results/replication/tables/exhibit_{exhibit}.csv")
    # The printed paper's rounded Exhibit 4 is also consistent with linearly
    # scaling its average dimensional metrics. Preserve that display diagnostic
    # separately from actual CAGR/drawdown recomputed on scaled daily returns.
    linear=pd.read_csv(ROOT/'results/replication/tables/exhibit_3.csv').set_index('strategy')
    factors=cfg['target_volatility']/linear.annual_volatility
    for column in ['annual_return','annual_volatility','downside_deviation','max_drawdown']:
        linear[column]*=factors
    linear.to_csv(ROOT/'results/replication/tables/exhibit_4_linear_display.csv')
    pd.DataFrame(costs).to_csv(ROOT/"results/replication/tables/transaction_costs_by_fold.csv",index=False)
    pd.DataFrame(costs).query("fold < 2020").groupby(["strategy","cost_bps"]).mean(numeric_only=True).drop(columns="fold").to_csv(ROOT/"results/replication/tables/transaction_costs.csv")
    pd.DataFrame(extension).to_csv(ROOT/"results/extension/tables/roll_costs_by_fold.csv",index=False)
    pd.DataFrame(extension).query("fold < 2020").groupby(["strategy","scenario","cost_bps"]).mean(numeric_only=True).drop(columns="fold").to_csv(ROOT/"results/extension/tables/roll_costs.csv")
    pd.DataFrame(years).to_csv(ROOT/"results/replication/tables/annual_metrics.csv",index=False)
    # Compare every requested model even if its experiment is unfinished.
    published=pd.read_csv(ROOT/"configs/published_exhibits.csv")
    reproduced=pd.read_csv(ROOT/"results/replication/tables/exhibit_3.csv")
    comparison=published.query("exhibit == 3").merge(reproduced,on="strategy",how="left",suffixes=("_published","_reproduced"))
    comparison["sharpe_difference"]=comparison.sharpe_reproduced-comparison.sharpe_published
    comparison["status"]=np.where(comparison.sharpe_reproduced.isna(),"not executed",
        np.where(comparison.test_windows.eq(5),"executed WRDS adaptation","partial test-window coverage"))
    comparison.to_csv(ROOT/"results/replication/tables/published_comparison.csv",index=False)
    diagnostics(p,daily,cfg)
    return daily

def diagnostics(p,daily,cfg):
    core=daily[daily.fold.lt(2020)]
    breakeven=[]
    for strategy,g in core.groupby("strategy"):
        breakeven.append({"strategy":strategy,"paper_break_even_bps":1e4*g.gross.sum()/g.paper_turnover.sum(),
            "raw_roll_break_even_bps":1e4*g.tradable_gross.sum()/g.roll_turnover.sum(),
            "annual_roll_turnover_extra":252*g.roll_extra.mean()})
    pd.DataFrame(breakeven).to_csv(ROOT/"results/extension/tables/break_even_costs.csv",index=False)
    q=p.copy();slow=q.norm_annual_return;fast=q.norm_monthly_return
    q["regime"]=np.select([(slow>=0)&(fast>=0),(slow<0)&(fast<0),(slow>=0)&(fast<0)],
        ["Bull","Bear","Correction"],default="Rebound")
    q["captured"]=q.position*q.target_returns
    q.groupby(["strategy","regime"]).agg(observations=("position","size"),
        mean_position=("position","mean"),mean_daily_captured=("captured","mean"),
        captured_vol=("captured","std")).to_csv(ROOT/"results/extension/tables/regime_diagnostics.csv")
    # Paired moving-block bootstrap inside each OOS fold. Conditional on the fitted
    # model; it does not measure retraining or hyperparameter-search uncertainty.
    pivot=core.pivot(index=["fold","label_date"],columns="strategy",values="gross")
    rng=np.random.default_rng(cfg["seed"]);rows=[]
    def sr(x):return np.mean(x)/np.std(x,ddof=1)*np.sqrt(252)
    for strategy in [s for s in pivot if s.startswith("LSTM-CPD")]:
        pairs=[g[["LSTM",strategy]].dropna().to_numpy() for _,g in pivot.groupby(level=0)]
        draws=[]
        for _ in range(500):
            diffs=[]
            for x in pairs:
                n=len(x);starts=rng.integers(0,n,size=int(np.ceil(n/63)))
                ix=((starts[:,None]+np.arange(63))%n).ravel()[:n]
                diffs.append(sr(x[ix,1])-sr(x[ix,0]))
            draws.append(np.mean(diffs))
        point=np.mean([sr(x[:,1])-sr(x[:,0]) for x in pairs])
        rows.append({"strategy":strategy,"difference_vs_lstm":point,
            "ci_2.5":np.quantile(draws,.025),"ci_97.5":np.quantile(draws,.975),
            "bootstrap_samples":500,"block_sessions":63})
    pd.DataFrame(rows).to_csv(ROOT/"results/extension/tables/paired_sharpe_bootstrap.csv",index=False)
