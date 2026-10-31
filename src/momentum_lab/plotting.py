"""Publication-style figures generated exclusively from completed experiments."""
import numpy as np
import pandas as pd
import os
from .paths import ROOT
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"results/replication/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={"Long Only":"#87939b","MACD":"#b99755","TSMOM w=0":"#d18c34",
    "TSMOM w=0.5":"#c3a77e","TSMOM w=1":"#a3afb3","LSTM":"#2769a0",
    "LSTM-CPD-10":"#a393c2","LSTM-CPD-21":"#1c997b","LSTM-CPD-63":"#7d5ba6",
    "LSTM-CPD-126":"#9a595a","LSTM-CPD-252":"#53695d","LSTM-CPD-Optimised":"#172e43"}

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.grid":True,"grid.alpha":.16,"figure.dpi":130,
        "savefig.dpi":180,"axes.titleweight":"bold","axes.labelcolor":"#293643"})

def save(fig,name,extension=False):
    directory=ROOT/("results/extension/figures" if extension else "results/replication/figures")
    directory.mkdir(parents=True,exist_ok=True);fig.savefig(directory/(name+".png"),bbox_inches="tight")
    fig.savefig(directory/(name+".pdf"),bbox_inches="tight");plt.close(fig)

def make_figures():
    style();d=pd.read_parquet(ROOT/"results/replication/daily_returns.parquet")
    preferred=["Long Only","TSMOM w=0","LSTM","LSTM-CPD-21","LSTM-CPD-63","LSTM-CPD-Optimised"]
    shown=[s for s in preferred if s in set(d.strategy)]
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.4))
    for s in shown:
        g=d[d.strategy.eq(s)].sort_values("label_date");r=g.set_index("label_date").gross
        wealth=(1+r).cumprod();axs[0].plot(wealth.index,wealth,label=s,color=COLORS[s],lw=1.2)
        average_vol=g[g.fold.lt(2020)].groupby('fold').gross.std().mean()*np.sqrt(252)
        rr=r*.15/average_vol;wealth=(1+rr).cumprod()
        axs[1].plot(wealth.index,wealth,label=s,color=COLORS[s],lw=1.2)
    axs[0].set(title="Raw signal output",ylabel="Growth of $1")
    axs[1].set(title="Ex-post 15% average-volatility rescaling",ylabel="Growth of $1 (log scale)",yscale="log")
    axs[0].legend(frameon=False,fontsize=8);fig.suptitle("Paper 67 — WRDS adaptation, out-of-sample",x=.51,y=1.02)
    save(fig,"wealth_comparison")
    fig,ax=plt.subplots(figsize=(10.5,3.7))
    for s in shown:
        r=d[d.strategy.eq(s)].sort_values("label_date").set_index("label_date").gross
        w=(1+r).cumprod();dd=w/w.cummax().clip(lower=1)-1
        ax.plot(dd.index,dd,label=s,color=COLORS[s],lw=1)
    ax.set(title="Drawdowns of raw signals",ylabel="Drawdown");ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(frameon=False,fontsize=8,ncol=3);save(fig,"drawdowns")
    cost=pd.read_csv(ROOT/"results/replication/tables/transaction_costs.csv")
    fig,ax=plt.subplots(figsize=(9,4.3))
    for s,g in cost.groupby("strategy"):
        ax.plot(g.cost_bps,g.sharpe,marker="o",ms=3,label=s,color=COLORS.get(s))
    ax.axvline(2.5,color="#bd5146",ls="--",lw=1);ax.axhline(0,color="gray",lw=.8)
    ax.set(title="Appendix C — turnover-adjusted Sharpe",xlabel="One-way cost (basis points of traded notional)",ylabel="Mean fold Sharpe")
    ax.legend(frameon=False,fontsize=8,bbox_to_anchor=(1,1));save(fig,"transaction_costs")
    comp=pd.read_csv(ROOT/"results/replication/tables/published_comparison.csv")
    fig,ax=plt.subplots(figsize=(11,4.3));x=np.arange(len(comp));width=.38
    ax.bar(x-width/2,comp.sharpe_published,width,label="Published (Pinnacle)",color="#a9b5bf")
    ax.bar(x+width/2,comp.sharpe_reproduced,width,label="Executed (WRDS)",color="#2769a0")
    ax.set_xticks(x,comp.strategy.str.replace("LSTM-CPD-","CPD "),rotation=42,ha="right")
    ax.set(title="Exhibit 3 — published and reproduced Sharpe",ylabel="Mean fold Sharpe");ax.legend(frameon=False)
    save(fig,"published_comparison")
    window_names=[f"LSTM-CPD-{l}" for l in [10,21,63,126,252]]
    indexed=comp.set_index("strategy")
    fig,ax=plt.subplots(figsize=(8.5,4))
    windows=[10,21,63,126,252]
    ax.plot(windows,indexed.loc[window_names,"sharpe_published"],marker="o",color="#a1adb8",label="Published (Pinnacle)")
    ax.plot(windows,indexed.loc[window_names,"sharpe_reproduced"],marker="o",color="#1c997b",label="Executed (WRDS)")
    if "LSTM" in indexed.index:
        ax.axhline(indexed.loc["LSTM","sharpe_reproduced"],ls="--",color="#2769a0",label="WRDS LSTM without CPD")
    ax.set(title="Exhibit 2 — CPD lookback sensitivity",xlabel="CPD lookback (sessions)",ylabel="Mean fold Sharpe",xticks=windows)
    ax.legend(frameon=False,fontsize=9);save(fig,"cpd_window_sensitivity")
    fold=pd.read_csv(ROOT/"results/replication/tables/exhibits_by_fold.csv").query("exhibit == 3")
    pivot=fold.pivot(index="strategy",columns="fold",values="sharpe")
    fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(pivot,aspect="auto",cmap="RdYlGn",vmin=-1,vmax=2)
    ax.set_xticks(np.arange(len(pivot.columns)),pivot.columns);ax.set_yticks(np.arange(len(pivot)),pivot.index)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            if np.isfinite(pivot.iloc[i,j]):ax.text(j,i,f"{pivot.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
    ax.set(title="Regime sensitivity — OOS Sharpe by test-window start",xlabel="2020 is a separate one-year holdout")
    fig.colorbar(im,ax=ax,label="Sharpe");save(fig,"regime_sensitivity")
    ext=pd.read_csv(ROOT/"results/extension/tables/roll_costs.csv")
    fig,axs=plt.subplots(1,3,figsize=(15,4),sharey=True)
    labels={"raw returns plus rolls":"Raw returns + rolls","one-session signal delay plus rolls":"Signal delay + rolls","2x exposure cap plus rolls":"2x exposure cap + rolls"}
    for ax,(scenario,g) in zip(axs,ext.groupby("scenario")):
        for s in shown:
            sub=g[g.strategy.eq(s)];ax.plot(sub.cost_bps,sub.sharpe,label=s,color=COLORS[s],marker="o",ms=3)
        ax.axvline(2.5,color="#bd5146",ls="--",lw=1);ax.axhline(0,color="gray",lw=.8)
        ax.set(title=labels[scenario],xlabel="One-way cost (bps)")
    axs[0].set_ylabel("Mean fold net Sharpe");axs[-1].legend(frameon=False,fontsize=8,bbox_to_anchor=(1,1))
    save(fig,"roll_and_execution_costs",extension=True)
    # Position diagnostics use actual frozen predictions, never a refit to the event.
    p=pd.read_parquet(ROOT/"results/replication/asset_predictions.parquet")
    fig,axs=plt.subplots(2,1,figsize=(11,5.3))
    for ax,(ticker,lo,hi,title) in zip(axs,[("SP","2007-01-01","2010-01-01","S&P 500: financial crisis"),
        ("BN","2015-01-01","2017-12-31","British pound: Brexit")]):
        for s in ["LSTM","LSTM-CPD-21","LSTM-CPD-63"]:
            q=p[p.ticker.eq(ticker)&p.strategy.eq(s)&p.date.between(lo,hi)]
            if len(q):ax.plot(q.date,q.position,label=s,color=COLORS[s],lw=.7,alpha=.85)
        ax.set(title=title,ylabel="Position",ylim=(-1.1,1.1))
    axs[0].legend(frameon=False,ncol=3,fontsize=8);fig.tight_layout();save(fig,"position_diagnostics")
