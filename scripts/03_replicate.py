"""Print audited official run instructions; executing requires a separate legacy environment."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['LSTM','LSTM-CPD-21','TFT','TFT-CPD-126-21'],default='TFT')
    a=p.parse_args()
    print(json.dumps({'status':'NOT_TRAINED','model':a.model,
        'upstream':str(ROOT/'vendor/trading-momentum-transformer'),
        'next_step':'Read docs/REPLICATION.md. Do not run upstream CLI unchanged for five-year paper folds.',
        'needs':['licensed/audited futures data','legacy TF/GPflow environment',
                 'CPD features if selected','corrected fold adapter and train-only preprocessing audit']},indent=2))
