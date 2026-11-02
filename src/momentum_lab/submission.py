"""Export aggregate results and regenerate the submitted research figures."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]


def export():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    source = ROOT/'results/author_five_assets/tables'
    tables = ROOT/'report/tables/replication'
    figures = ROOT/'report/figures/replication'
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    if source.exists():
        for path in source.iterdir():
            if path.is_file():
                name = 'roll_cost_sensitivity.csv' if path.name=='professor_roll_cost_sensitivity.csv' else path.name
                shutil.copy2(path, tables/name)
        manifest = ROOT/'report/author_five_assets/execution_manifest.json'
        if manifest.exists():
            shutil.copy2(manifest, ROOT/'report/execution_manifest.json')
    ours = pd.read_csv(tables/'exhibit_3.csv').set_index('strategy')
    paper = pd.read_csv(tables/'published_reference.csv')
    paper = paper[paper.exhibit.eq(3)].set_index('strategy')
    comparison = ours[['sharpe_ratio']].rename(columns={'sharpe_ratio':'Five-instrument WRDS Sharpe'}).join(
        paper[['sharpe']].rename(columns={'sharpe':'Paper 50-instrument Pinnacle Sharpe'}))
    comparison.to_csv(ROOT/'report/paper_reference_comparison.csv')
    order = ['LSTM','LSTM-CPD-10','LSTM-CPD-21','LSTM-CPD-63','LSTM-CPD-126','LSTM-CPD-252','LSTM-CPD-Optimised']
    ax = comparison.loc[order].plot.bar(figsize=(12,5),rot=25,color=['#237b83','#afb9c4'])
    ax.set_ylabel('Mean test-window gross Sharpe')
    ax.set_title('Five-asset study and published reference (different universes)')
    ax.figure.tight_layout()
    ax.figure.savefig(figures/'paper_comparison.png',dpi=160)
    plt.close(ax.figure)
    folds = pd.read_csv(tables/'exhibits_by_fold.csv')
    values = folds[folds.strategy.isin(['LSTM','LSTM-CPD-Optimised'])].pivot(index='fold',columns='strategy',values='sharpe_ratio')
    ax = values.plot.bar(figsize=(10,4),rot=0)
    ax.set_title('Five-asset LSTM and validation-selected CPD by test period')
    ax.set_ylabel('Gross Sharpe')
    ax.figure.tight_layout()
    ax.figure.savefig(figures/'fold_comparison.png',dpi=160)
    plt.close(ax.figure)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax, name, title in zip(axes,['author_transaction_costs.csv','roll_cost_sensitivity.csv'],
            ['Author cost convention','Raw returns plus both roll legs']):
        data = pd.read_csv(tables/name)
        data = data[data.strategy.isin(['LSTM','LSTM-CPD-Optimised'])].groupby(['strategy','bps']).sharpe_ratio.mean().unstack('strategy')
        data.plot(ax=ax,marker='o')
        ax.axhline(0,color='gray',lw=.7)
        ax.set_title(title)
        ax.set_xlabel('One-way transaction cost (bps)')
        ax.set_ylabel('Mean test-window net Sharpe')
    fig.tight_layout()
    fig.savefig(figures/'transaction_costs.png',dpi=160)
    plt.close(fig)
    path = ROOT/'report/result_artifacts_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest['files'] = {p.relative_to(ROOT).as_posix():{
        'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for folder in [tables,figures] for p in folder.rglob('*') if p.is_file()}
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Exported aggregate tables, comparison and figures under report/.')
