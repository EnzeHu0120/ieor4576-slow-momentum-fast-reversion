"""One command rebuilds a labelled TSMOM demo or evaluates an audited return panel."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from momentum_lab.data import synthetic_returns,load_returns
from momentum_lab.features import tsmom
from momentum_lab.backtest import run_backtest
from momentum_lab.evaluation import metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--demo',action='store_true');g.add_argument('--returns',type=Path)
    p.add_argument('--contracts',type=Path);p.add_argument('--start',default='2012-01-01')
    p.add_argument('--config',type=Path,default=ROOT/'configs/pilot.yaml')
    a=p.parse_args();r=synthetic_returns() if a.demo else load_returns(a.returns)
    contracts=pd.read_parquet(a.contracts).reindex(index=r.index,columns=r.columns) if a.contracts else None
    c=yaml.safe_load(a.config.read_text());signal=tsmom(r,c['lookback'])
    out=ROOT/'results'/('demo' if a.demo else 'pilot');out.mkdir(parents=True,exist_ok=True)
    rows=[];fig,ax=plt.subplots(figsize=(9,4))
    for bps in c['cost_bps']:
        bt=run_backtest(r,signal,bps,contracts,c['execution_lag'],c['target_vol'],c['max_asset_exposure'])
        s=bt['net'].loc[a.start:];rows.append({'cost_bps':bps,**metrics(s,bt['turnover'])})
        ax.plot((1+s).cumprod(),label=f'{bps:g} bps')
        pd.DataFrame({'gross':bt['gross'],'net':bt['net'],'turnover':bt['turnover']}).to_csv(out/f'daily_{bps:g}bps.csv')
    pd.DataFrame(rows).to_csv(out/'cost_sensitivity.csv',index=False)
    ax.set(title='Synthetic plumbing demo' if a.demo else 'TSMOM pilot — substitute data',ylabel='Wealth (start = 1)')
    ax.legend(ncol=3);fig.tight_layout();fig.savefig(out/'equity.png',dpi=160);plt.close(fig)
    (out/'run.json').write_text(json.dumps({'source':'SYNTHETIC' if a.demo else str(a.returns),
        'config':c,'evaluation_start':a.start,'status':'pipeline_demo' if a.demo else 'pilot_not_paper_replication',
        'contract_roll_costs':contracts is not None,
        'terminal_liquidation':False,'cash_interest':False,'model':'TSMOM_252'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(out)
