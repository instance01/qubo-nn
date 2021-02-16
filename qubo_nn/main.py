from qubo_nn.pipeline import Classification
from qubo_nn.config import Config
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("type")
parser.add_argument("cmd")
parser.add_argument("cfg_id")
parser.add_argument("--model", nargs="?")
args = parser.parse_args()

cfg = Config().get_cfg(args.cfg_id)

if args.type == 'classify':
    print(args.cmd)
    if args.cmd == 'gendata':
        Classification(cfg).prep_data()
    elif args.cmd == 'train':
        Classification(cfg).run_experiment()
    elif args.cmd == 'eval':
        Classification(cfg).eval(args.model)
