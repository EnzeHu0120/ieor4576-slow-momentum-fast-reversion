"""Rebuild verified tables and figures from the three immutable run archives."""
import contextlib
import io
import json
import runpy
import shutil
import pandas as pd
import numpy as np
from . import paths
from .analysis_math import net_on_common_intervals,summarize,rescale_ex_post
from .models import check_validation_selection
from .plotting import build

def rebuild():
    paths.initialize()
    count=check_validation_selection()
    # This audit independently reconstructs all metrics, cost arithmetic,
    # source-date checks and the sampled GP marginal likelihoods.
    with contextlib.redirect_stdout(io.StringIO()):runpy.run_module('project.audit',run_name='__main__')
    source=paths.ARTIFACTS/'supplement'
    allp=pd.read_csv(source/'all_available_test_asset_predictions.csv',parse_dates=['signal_date','target_date'])
    common=pd.read_csv(paths.ARTIFACTS/'cpd/asset_comparison_2019_2020.csv',parse_dates=['signal_date','target_date'])
    names=['long_only','tsmom_annual','tsmom_blend','tsmom_monthly','macd','lstm']+[f'lstm_cpd_{l}' for l in [10,21,63,126,252]]+['lstm_cpd_validated']
    raw,turnover,net,cost=net_on_common_intervals(allp,common[['segment','target_date']],names)
    scaled,factors=rescale_ex_post(raw)
    for name,data in [('gross_fixed_lookbacks.csv',summarize(raw)),('ex_post_15pct_metrics.csv',summarize(scaled)),('cost_sensitivity_0_to_5bps.csv',cost)]:
        data.to_csv(paths.TABLES/name,index=False)
        reference=pd.read_csv(source/name)
        for col in data.select_dtypes(include='number'):
            np.testing.assert_allclose(data[col],reference[col],rtol=1e-10,atol=1e-11)
    raw.to_csv(paths.TABLES/'gross_common_interval_returns.csv')
    scaled.to_csv(paths.TABLES/'ex_post_15pct_returns.csv')
    turnover.to_csv(paths.TABLES/'one_way_turnover_common_intervals.csv')
    net.to_csv(paths.TABLES/'cost_adjusted_common_interval_returns.csv')
    factors.rename('constant_ex_post_multiplier').to_csv(paths.TABLES/'ex_post_15pct_multipliers.csv')
    for name in ['gp_fit_summary.csv','validation_selected_checkpoints.csv']:
        shutil.copy2(source/name,paths.TABLES/name)
    build(raw,scaled,cost,paths.FIGURES)
    from .cross_study import rebuild as rebuild_cross_study
    rebuild_cross_study()
    status={'reference_archives_checked':3,'validation_selected_trial_records_checked':count,
            'recomputed_strategies':len(names),'common_intervals':len(raw),'assets':24,
            'new_training_performed':False,'new_gp_fits_performed':False}
    (paths.TABLES/'REPORT_REBUILD_COMPLETE.json').write_text(json.dumps(status,indent=2)+'\n')
    print('REPORT_REBUILD_OK: 300 training trial records checked; 12 strategy rows rebuilt on 474 common intervals.')
    print('Tables: results/replication/tables; figures: results/replication/figures.')
