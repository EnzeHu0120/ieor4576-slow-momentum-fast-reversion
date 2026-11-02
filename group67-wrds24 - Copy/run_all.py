"""Thin checkout entry point; editable installation is optional."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from project.cli import main

if __name__=='__main__':main()
