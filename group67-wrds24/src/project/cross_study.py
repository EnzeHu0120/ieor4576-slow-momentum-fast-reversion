"""Descriptive cross-study context; not a controlled market-count experiment."""
import json
import pandas as pd
import numpy as np
from . import paths


def rebuild():
    paths.initialize()
    evidence=paths.ROOT/'report/evidence'
    five=evidence/'enze_five_assets'
    folds=pd.read_csv(five/'exhibits_by_fold.csv')
    reference=pd.read_csv(five/'paper_reference_comparison.csv')
    means=folds.groupby('strategy').sharpe_ratio.mean()
    for row in reference.itertuples(index=False,name=None):
        if not np.isclose(means[row[0]],float(row[1]),rtol=0,atol=1e-12):
            raise ValueError('Five-asset fold mean differs from its committed summary: '+row[0])
    selected=pd.read_csv(five/'selected_cpd_windows.csv')
    a=folds[folds.strategy.eq('LSTM')][['fold','sharpe_ratio']].rename(columns={'sharpe_ratio':'lstm_sharpe'})
    b=folds[folds.strategy.eq('LSTM-CPD-Optimised')][['fold','sharpe_ratio']].rename(columns={'sharpe_ratio':'selected_cpd_sharpe'})
    comparison=a.merge(b,on='fold',validate='one_to_one').merge(selected[['fold','strategy']],on='fold',validate='one_to_one')
    if len(comparison)!=5:raise ValueError('Expected five independently reported test windows.')
    comparison['cpd_minus_lstm_sharpe']=comparison.selected_cpd_sharpe-comparison.lstm_sharpe
    for row in comparison.itertuples():
        fixed=folds[folds.fold.eq(row.fold)&folds.strategy.eq(row.strategy)]
        if len(fixed)!=1 or not np.isclose(fixed.sharpe_ratio.iloc[0],row.selected_cpd_sharpe,atol=1e-12,rtol=0):
            raise ValueError('The five-asset selected model differs from its fixed-window output.')
    pilot=json.loads((evidence/'pilot_panel_observation.json').read_text())['observations'][0]
    panel=pd.read_csv(paths.TABLES/'gross_fixed_lookbacks.csv').set_index('strategy')
    items=[
       {'study':'Single-sequence development pilot','markets':1,'lstm_sharpe':pilot['ordinary_lstm_sharpe'],
        'selected_cpd_sharpe':pilot['selected_cpd_sharpe'],'test_period':'2019-2020','aggregation':'One 505-observation sequence',
        'data_construction':'WRDS code 1825; instrument/roll identity unverified'},
       {'study':'Enze five-asset author-source study','markets':5,'lstm_sharpe':means['LSTM'],
        'selected_cpd_sharpe':means['LSTM-CPD-Optimised'],'test_period':'1995-2020',
        'aggregation':'Unweighted mean of five test-window Sharpe ratios',
        'data_construction':'WRDS individual contracts; same-held-contract returns and explicit rolls'},
       {'study':'Current 24-market adaptation','markets':24,'lstm_sharpe':panel.loc['lstm','sharpe_ratio'],
        'selected_cpd_sharpe':panel.loc['lstm_cpd_validated','sharpe_ratio'],'test_period':'2019-2020',
        'aggregation':'One sequence of 474 retained common intervals',
        'data_construction':'WRDS vendor continuous price indices; provisional underlying mappings'},
    ]
    for item in items:
        item['cpd_minus_lstm_sharpe']=item['selected_cpd_sharpe']-item['lstm_sharpe']
        item['controlled_universe_size_experiment']=False
    pd.DataFrame(items).to_csv(paths.TABLES/'cross_study_context.csv',index=False)
    comparison.to_csv(paths.TABLES/'five_asset_fold_comparison.csv',index=False)
    (paths.TABLES/'CROSS_STUDY_CHECK.json').write_text(json.dumps({
        'five_asset_commit':'f88874663ba3641f06c867c45446b1c4e5faf93e',
        'fold_means_match_committed_summary':True,
        'five_asset_folds_with_cpd_above_lstm':int(comparison.cpd_minus_lstm_sharpe.gt(0).sum()),
        'five_asset_total_folds':5,
        'five_asset_mean_difference':float(means['LSTM-CPD-Optimised']-means['LSTM']),
        'evidence_scope':'Checks committed aggregate results; five-asset raw quotes, weights and individual positions were not independently replayed.',
        'universe_size_causal_effect_established':False,
        'new_experiments_performed':False},indent=2)+'\n')


if __name__=='__main__':rebuild()
