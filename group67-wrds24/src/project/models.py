"""Train the audited model family or inspect its validation selections."""
import json
import math
import numpy as np
import pandas as pd
from . import paths
from .features import config,execute

def fit(family):
    c=config();folder='lstm_seed67' if family=='lstm' else 'cpd_seed67'
    args=['train','--model',family,'--device',c['device'],'--seed',c['seed'],
          '--trials',c['trials_per_variant'],'--epochs',c['max_epochs'],'--patience',c['patience'],
          '--output','results/wrds_panel24/'+folder]
    if family=='cpd':args+=['--frozen-lstm','results/wrds_panel24/lstm_seed67']
    execute('run_wrds_panel24.py',*args)

def supplement():
    c=config();execute('run_wrds_panel24_analysis.py','--device',c['device'],
                      '--frozen-lstm','results/wrds_panel24/lstm_seed67')

def check_validation_selection():
    roots=[paths.ARTIFACTS/'lstm',paths.ARTIFACTS/'cpd']
    expected=50;count=0
    for root in roots:
        cfg=json.loads((root/'run_config.json').read_text())
        variants=['lstm'] if root.name=='lstm' else ['cpd_'+str(x) for x in config()['cpd_lookbacks']]
        winners=[]
        for variant in variants:
            records=[json.loads((root/variant/f'trial_{i:03d}/status.json').read_text()) for i in range(1,expected+1)]
            if not all(r['status']=='completed' for r in records):raise ValueError('An experiment trial failed.')
            for i,record in enumerate(records,1):
                if record['parameters']!=cfg['candidates'][i-1] or record['seed']!=cfg['seed']+1009*i:
                    raise ValueError('Trial parameters or seed mismatch.')
                history=pd.read_csv(root/variant/f'trial_{i:03d}/history.csv')
                if len(history)!=record['epochs_ran'] or history.epoch.tolist()!=list(range(1,len(history)+1)):
                    raise ValueError('Incomplete trial history.')
                if not np.isfinite(history[['loss','validation_sharpe']]).all().all():raise ValueError('Non-finite training history.')
                batches=math.ceil(record['training_sequences']/record['parameters']['batch_size'])
                if history.optimizer_steps.tolist()!=[batches*e for e in history.epoch]:raise ValueError('Update count mismatch.')
                best=-np.inf;best_epoch=wait=0
                for epoch in history.itertuples():
                    if epoch.validation_sharpe>best+1e-4:best=epoch.validation_sharpe;best_epoch=epoch.epoch;wait=0
                    else:wait+=1
                if not np.isclose(best,record['validation_sharpe'],rtol=0,atol=1e-10) or best_epoch!=record['best_epoch']:
                    raise ValueError('Early-stopping checkpoint mismatch.')
                if len(history)!=cfg['epochs'] and wait!=cfg['patience']:raise ValueError('Patience rule mismatch.')
                if record['optimizer_steps_in_best_checkpoint']!=best_epoch*batches:raise ValueError('Selected checkpoint step mismatch.')
            winner=max(records,key=lambda r:r['validation_sharpe'])
            if winner!=json.loads((root/variant/'best_validation.json').read_text()):raise ValueError('Variant selection mismatch.')
            winners.append(winner);count+=len(records)
        chosen=json.loads((root/'SELECTED_USING_VALIDATION_ONLY.json').read_text())
        if max(winners,key=lambda r:r['validation_sharpe'])!=chosen['lstm' if root.name=='lstm' else 'lstm_cpd']:
            raise ValueError('Overall validation-only selection mismatch.')
    return count
