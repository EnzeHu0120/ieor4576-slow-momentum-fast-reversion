"""Print portable compute readiness without reading credentials or connecting to WRDS."""
import json
import platform
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from momentum_lab.models import resolve_device

if __name__=='__main__':
    cuda=torch.cuda.is_available()
    print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
        'cuda_available':cuda,'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if cuda else None,
        'metal_available':torch.backends.mps.is_available(),
        'selected_device':str(resolve_device())},indent=2))
