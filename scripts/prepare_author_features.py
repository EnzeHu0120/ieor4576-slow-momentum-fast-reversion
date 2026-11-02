from pathlib import Path
import os,sys
os.environ['TF_USE_LEGACY_KERAS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from momentum_lab.author_native import configuration,prepare_features
if __name__=='__main__':prepare_features(configuration('configs/author_five_assets.yaml'))
