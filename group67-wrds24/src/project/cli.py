"""One command rebuilds the reference report; full estimation is explicit."""
import argparse
from .data import import_archives,import_raw,import_new_runtime_results
from .evaluation import rebuild
from .features import build_base,build_cpd
from .models import fit,supplement

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['report','full'],default='report')
    parser.add_argument('--bundle-dir',help='Directory containing the three previously saved result ZIP files.')
    parser.add_argument('--author-repo',help='Existing author repository from which to copy private raw WRDS inputs.')
    parser.add_argument('--workers',type=int,default=None)
    args=parser.parse_args()
    if args.author_repo:import_raw(args.author_repo)
    if args.stage=='report':
        import_archives(args.bundle_dir);rebuild()
    else:
        # Expensive, optional full re-estimation. Preserve its outputs apart
        # from the original verified experiment because training can vary.
        build_base();fit('lstm');build_cpd(args.workers);fit('cpd');supplement()
        import_new_runtime_results()

if __name__=='__main__':main()
