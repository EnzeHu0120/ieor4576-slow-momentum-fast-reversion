from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from momentum_lab.progress import status
if __name__=='__main__':print(json.dumps(status()))
