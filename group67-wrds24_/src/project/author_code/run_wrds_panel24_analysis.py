"""Evaluate existing checkpoints; do not fit GPs or train any model."""
import argparse
import gc
import json
import shutil
import zipfile
from pathlib import Path
from paper67_panel_runtime import bootstrap,digest,require,save_json


def resolve_frozen_lstm(run,cfg,explicit=None):
    """Support both the live training tree and the exported archive layout."""
    candidates=[Path(explicit)] if explicit else [run/'frozen_lstm',run.parent/'lstm_seed67']
    required=['run_config.json','RUN_COMPLETE.json','SELECTED_USING_VALIDATION_ONLY.json','best_lstm.weights.h5']
    expected=cfg.get('frozen_lstm_config_sha256')
    require(expected,'The CPD run does not identify its frozen ordinary LSTM configuration.')
    for candidate in candidates:
        if not all((candidate/name).is_file() for name in required):continue
        if digest(candidate/'run_config.json')==expected:return candidate
    raise ValueError('Cannot find the matched ordinary LSTM run. Checked: '+
                     ', '.join(str(p) for p in candidates)+
                     '. Provide --frozen-lstm with the original completed LSTM output directory. No retraining is needed.')


def audit_gp(original,frame,output):
    import numpy as np
    import pandas as pd
    path=original.RESULTS/'cpd/CPD_COMPLETE.json'
    complete=json.loads(path.read_text())
    cfg=complete['config']
    require(complete['cpd_features_sha256']==digest(original.CPD_PATH),'CPD feature hash mismatch.')
    require(cfg['features_sha256']==digest(original.FEATURE_PATH),'Base feature hash mismatch.')
    require(cfg['levels_sha256']==digest(original.LEVEL_PATH),'Segmented price hash mismatch.')
    require(cfg['author_cpd_sha256']==digest(Path('mom_trans/changepoint_detection.py')),'Author CPD source mismatch.')
    require(complete['windows']==len(frame)*5,'Unexpected GP window count.')
    base=original.read_table(original.FEATURE_PATH)
    require(frame[['segment','signal_date','target_date']].equals(base[['segment','signal_date','target_date']]),'Feature row ordering changed.')
    for col in base.select_dtypes(include='number'):
        np.testing.assert_allclose(frame[col],base[col],rtol=1e-12,atol=1e-12)
    diagnostic=pd.read_csv(original.RESULTS/'cpd/fit_windows.csv',parse_dates=['date'])
    require(len(diagnostic)==len(frame)*5,'Incomplete fit diagnostics.')
    require(not diagnostic.duplicated(['segment','lookback','date']).any(),'Duplicate GP fit records.')
    require(set(diagnostic.status)<=set(['completed','failed']),'Unexpected fit status.')
    summaries=[];samples=[]
    for lb in original.LOOKBACKS:
        columns=['segment','signal_date','source_position',f'cp_score_{lb}',f'cp_rl_{lb}',f'cp_source_date_{lb}']
        f=frame[columns].merge(diagnostic[diagnostic.lookback.eq(lb)],left_on=['segment','signal_date'],right_on=['segment','date'],validate='one_to_one',how='left')
        require(f.status.notna().all(),'A GP window diagnostic is missing.')
        f=f.sort_values(['segment','source_position']).reset_index(drop=True)
        ok=f.status.eq('completed')
        require(f.loc[ok,f'cp_score_{lb}'].between(-1e-8,1+1e-8).all(),'Invalid GP score.')
        require(f.loc[ok,f'cp_rl_{lb}'].between(-1e-8,1+1e-8).all(),'Invalid completed GP location.')
        expected_date=f.signal_date.where(ok).groupby(f.segment).ffill()
        source=pd.to_datetime(f[f'cp_source_date_{lb}'])
        require(expected_date.notna().all() and source.eq(expected_date).all(),'GP fill does not use the latest past successful fit.')
        score=f[f'cp_score_{lb}'].where(ok).groupby(f.segment).ffill()
        anchor=(f.source_position-lb*f[f'cp_rl_{lb}']).where(ok).groupby(f.segment).ffill()
        np.testing.assert_allclose(f[f'cp_score_{lb}'],score,rtol=0,atol=1e-12)
        np.testing.assert_allclose(f[f'cp_rl_{lb}'],(f.source_position-anchor)/lb,rtol=1e-10,atol=1e-10)
        summaries.append({'lookback':lb,'windows':len(f),'failed_fits_filled_from_past':int((~ok).sum()),
                          'mean_fit_seconds':float(f.elapsed_seconds.mean())})
        for seg,g in f.groupby('segment'):
            for i in sorted(set([0,len(g)//2,len(g)-1])):
                r=g.iloc[i];date=r.signal_date.strftime('%Y-%m-%d')
                p=original.PROCESSED/'cpd_cache'/seg/f'lbw_{lb}'/(date+'.json')
                require(p.is_file(),'Sample GP cache file is missing: '+str(p))
                record=json.loads(p.read_text())
                require(record['segment']==seg and record['lookback']==lb and record['date']==date,'Sample cache key mismatch.')
                require(record['status']==r.status,'Sample cache status mismatch.')
                if record['status']=='completed':
                    np.testing.assert_allclose([r[f'cp_score_{lb}'],r[f'cp_rl_{lb}']],
                                              [record['cp_score'],record['cp_location_norm']],atol=1e-8,rtol=0)
                destination=output/'audit_inputs/cache_samples'/seg/f'lbw_{lb}'/p.name
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
                samples.append(str(destination.relative_to(output)))
    require(sum(s['failed_fits_filled_from_past'] for s in summaries)==len(complete['failed_fits_filled_from_past']),'Failure count mismatch.')
    pd.DataFrame(summaries).to_csv(output/'gp_fit_summary.csv',index=False)
    save_json(output/'GP_POSTCHECK.json',{'checks_passed':True,'window_count':len(diagnostic),
              'sample_cache_records':len(samples),'full_cache_refitted':False,'summary':summaries})
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='results/wrds_panel24/cpd_seed67')
    parser.add_argument('--output',default='results/wrds_panel24/supplement_seed67')
    parser.add_argument('--frozen-lstm',default=None,help='Optional original completed ordinary LSTM directory.')
    parser.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    args=parser.parse_args()
    bootstrap(args.device)
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import keras_tuner as kt
    import run_wrds_panel24 as original
    from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel
    from paper67_panel24_analysis import add_positions,net_on_common_intervals,summarize,rescale_ex_post,plot_results
    run=Path(args.run);out=Path(args.output)
    require(run.resolve()!=out.resolve(),'Use a separate analysis output directory.')
    require((run/'RUN_COMPLETE.json').is_file(),'The original CPD-LSTM run is incomplete.')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((run/'run_config.json').read_text())
    plain_root=resolve_frozen_lstm(run,cfg,args.frozen_lstm)
    print('Matched ordinary LSTM run:',plain_root,flush=True)
    require(digest(Path('run_wrds_panel24.py'))==cfg['runner_sha256'],'Original runner changed.')
    for relative,key in [('mom_trans/deep_momentum_network.py','author_model_sha256'),('mom_trans/model_inputs.py','author_batch_sha256')]:
        require(digest(Path(relative))==cfg[key],'Author source changed: '+relative)
    require(digest(original.CPD_PATH)==cfg['feature_sha256'],'CPD feature hash differs from the trained run.')
    frame=original.read_table(original.CPD_PATH)
    audit_gp(original,frame,out)
    common=original.read_table(run/'asset_comparison_2019_2020.csv')
    keys=['segment','target_date']
    full=None;selections=[];replay_errors={}
    winner=json.loads((run/'SELECTED_USING_VALIDATION_ONLY.json').read_text())['lstm_cpd']
    plain=json.loads((plain_root/'SELECTED_USING_VALIDATION_ONLY.json').read_text())['lstm']
    variants=[('lstm',plain,plain_root/'best_lstm.weights.h5')]
    for lb in original.LOOKBACKS:
        best=json.loads((run/f'cpd_{lb}/best_validation.json').read_text())
        variants.append((f'lstm_cpd_{lb}',best,run/f'cpd_{lb}'/f"trial_{best['trial']:03d}"/'best.weights.h5'))
    tf.config.set_soft_device_placement(True)
    gpus=tf.config.list_physical_devices('GPU')
    if args.device=='gpu':
        require(gpus,'GPU is not visible. Use the same GPU runtime as the original run.')
        for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    device='/GPU:0' if args.device=='gpu' else '/CPU:0'
    for name,best,checkpoint in variants:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(best['seed'])
        lb=best['lookback'];columns=original.BASE if lb is None else original.BASE+[f'cp_score_{lb}',f'cp_rl_{lb}']
        network=object.__new__(LstmDeepMomentumNetworkModel)
        network.time_steps,network.input_size,network.output_size=63,len(columns),1
        hp=kt.HyperParameters()
        for k,v in best['parameters'].items():
            if k!='batch_size':hp.Fixed(k,v)
        with tf.device(device):model=network.model_builder(hp)
        require(checkpoint.is_file(),'Missing saved checkpoint: '+str(checkpoint))
        model.optimizer.build(model.trainable_variables);model.load_weights(str(checkpoint))
        require(int(model.optimizer.iterations.numpy())==best['optimizer_steps_in_best_checkpoint'],'Checkpoint optimizer step mismatch.')
        if args.device=='gpu':require('GPU' in model.weights[0].device,'Model variables are not on GPU.')
        batch=original.batches(frame,columns,test=True)
        position=model.predict(batch['inputs'],verbose=0)[:,-1,0].astype(float)
        repeat=model.predict(batch['inputs'],verbose=0)[:,-1,0].astype(float)
        np.testing.assert_allclose(position,repeat,atol=1e-7,rtol=1e-6)
        require(np.isfinite(position).all() and (np.abs(position)<=1+1e-6).all(),'Invalid positions.')
        pred=pd.DataFrame({'segment':batch['identifier'][:,-1,0],
                           'target_date':pd.to_datetime(batch['date'][:,-1,0]),name+'_position':position})
        if full is None:
            full=pred.merge(frame,on=keys,how='left',validate='one_to_one')
            full=add_positions(full)
        else:full=full.merge(pred,on=keys,how='left',validate='one_to_one')
        require(full[name+'_position'].notna().all(),'Inference calendars differ between variants.')
        if name=='lstm' or lb==winner['lookback']:
            captured='lstm' if name=='lstm' else 'lstm_cpd'
            replay=common[keys+[captured+'_position']].merge(pred,on=keys,validate='one_to_one')
            error=float(np.max(np.abs(replay[captured+'_position']-replay[name+'_position']))) if name!=captured else float(np.max(np.abs(replay[captured+'_position_x']-replay[name+'_position_y'])))
            require(error<=1e-5,'Saved output replay differs by '+str(error)+'. Stop and send this log; do not replace reported results.')
            replay_errors[captured]=error
        selections.append({'strategy':name,'lookback':lb,'trial':best['trial'],
                           'validation_sharpe':best['validation_sharpe'],'best_epoch':best['best_epoch'],
                           'optimizer_steps':best['optimizer_steps_in_best_checkpoint'],
                           'checkpoint_sha256':digest(checkpoint)})
        print('Checkpoint evaluated:',name,'| validation Sharpe:',best['validation_sharpe'],flush=True)
        del model,batch;gc.collect()
    validated=f"lstm_cpd_{winner['lookback']}"
    full['lstm_cpd_validated_position']=full[validated+'_position']
    names=['long_only','tsmom_annual','tsmom_blend','tsmom_monthly','macd','lstm']+[f'lstm_cpd_{l}' for l in original.LOOKBACKS]+['lstm_cpd_validated']
    raw,turnover,net,cost_table=net_on_common_intervals(full,common[keys],names)
    for name,captured in [('lstm','lstm'),('lstm_cpd_validated','lstm_cpd')]:
        expected=common.groupby('target_date')[captured+'_author_gross'].sum()/24
        np.testing.assert_allclose(raw[name],expected,atol=1e-8,rtol=1e-5)
    scaled,factors=rescale_ex_post(raw)
    summarize(raw).to_csv(out/'gross_fixed_lookbacks.csv',index=False)
    summarize(scaled).to_csv(out/'ex_post_15pct_metrics.csv',index=False)
    factors.rename('constant_ex_post_multiplier').to_csv(out/'ex_post_15pct_multipliers.csv')
    raw.to_csv(out/'gross_common_interval_returns.csv');scaled.to_csv(out/'ex_post_15pct_returns.csv')
    turnover.to_csv(out/'one_way_turnover_common_intervals.csv')
    net.to_csv(out/'cost_adjusted_common_interval_returns.csv');cost_table.to_csv(out/'cost_sensitivity_0_to_5bps.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'validation_selected_checkpoints.csv',index=False)
    full.to_csv(out/'all_available_test_asset_predictions.csv',index=False)
    plot_results(raw,scaled,cost_table,out/'figures')
    for name in ['run_config.json','RUN_COMPLETE.json','SELECTED_USING_VALIDATION_ONLY.json']:
        dest=out/'audit_inputs/frozen_lstm'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(plain_root/name,dest)
    for p in [original.CPD_PATH,original.FEATURE_PATH,original.LEVEL_PATH,
              original.RESULTS/'cpd/CPD_COMPLETE.json',original.RESULTS/'cpd/fit_windows.csv',
              Path('mom_trans/changepoint_detection.py'),Path('paper67_panel_gp.py'),
              Path('run_wrds_panel24_cpd_safe.py'),Path('run_wrds_panel24_analysis.py'),Path('paper67_panel24_analysis.py')]:
        require(p.is_file(),'Required audit input missing: '+str(p))
        dest=out/'audit_inputs'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save_json(out/'ANALYSIS_COMPLETE.json',{'inference_device':device,'model_fits_performed':0,'gp_fits_performed':0,
              'saved_output_replay_max_position_errors':replay_errors,'assets':24,'common_intervals':len(raw),
              'full_available_test_prediction_rows':len(full),'selected_cpd_lookback_from_validation':winner['lookback'],
              'frozen_lstm_directory':str(plain_root),'frozen_lstm_config_sha256':digest(plain_root/'run_config.json'),
              'selection_changed_after_test':False,'risk_standardization':'Constant .15 / realized full-common-test volatility; ex-post presentation only, not a causal strategy.',
              'cost_policy':'Paper Eq.15: absolute changes in volatility-scaled exposure on all available per-segment asset observations BEFORE selecting the 474 common intervals; first available turnover is zero, matching author code.',
              'limitations':['Continuous-index transaction-cost proxy; futures rolls, contract sizing, bid-ask spreads and exchange-specific fees are not reconstructed.',
                             '474 matching intervals omit unmatched dates. Cost statistics remain conditional and do not establish full-calendar executable net performance.',
                             'Fixed-lookback OOS tables are diagnostics; preserve the winner already selected on 2017-2018 validation.',
                             'WRDS 24 assets, one split and one search seed; not the original 50-asset expanding-window experiment.']})
    bundle=out/'supplement_bundle.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=bundle:z.write(p,p.relative_to(out))
    print(summarize(raw)[['strategy','sharpe_ratio','max_drawdown']].to_string(index=False))
    print('Supplement bundle:',bundle)
    print('PANEL24_ANALYSIS_OK',flush=True)


if __name__=='__main__': main()
