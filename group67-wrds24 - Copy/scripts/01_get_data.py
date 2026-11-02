"""Thin entry point; logic lives in src/project."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from project.data import import_archives,import_raw
import argparse
p=argparse.ArgumentParser();p.add_argument('--bundle-dir');p.add_argument('--author-repo');a=p.parse_args()
if a.author_repo:import_raw(a.author_repo)
import_archives(a.bundle_dir)
