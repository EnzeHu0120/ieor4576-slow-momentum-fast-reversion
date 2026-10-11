"""Paper #67 readiness checker: LSTM / LSTM with online GP-CPD."""
import argparse, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--model",choices=["LSTM","LSTM-CPD-21","LSTM-CPD-63"],default="LSTM-CPD-21")
 a=p.parse_args()
 print(json.dumps({"paper":"Wood, Roberts & Zohren (2022), arXiv:2105.13727, catalog #67",
  "model":a.model,"status":"NOT_TRAINED",
  "source":str(ROOT/"vendor/author-lstm-cpd/mom_trans/deep_momentum_network.py"),
  "cpd_source":str(ROOT/"vendor/slow-momentum-fast-reversion/src/changepoint_detection.py"),
  "protocol":str(ROOT/"docs/REPLICATION.md"),
  "needs":["audited futures data and causal roll conventions",
    "isolated compatible TensorFlow/GPflow environment",
    "dated GP changepoint features for CPD model",
    "fixed chronological train/validation/test splits",
    "aligned gross and net evaluation across 0/1/2/3/5 bps"]},indent=2))
