"""Optional typesetting; empirical results are read rather than re-estimated."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from project.reporting import build_pdf

if __name__=='__main__':build_pdf()
