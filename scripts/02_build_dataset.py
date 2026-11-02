from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from momentum_lab.dataset import build
if __name__=='__main__':
    build('configs/data_wrds.yaml')
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_author_features.py')],check=True)
