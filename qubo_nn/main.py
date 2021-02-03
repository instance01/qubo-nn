from qubo_nn.pipeline import Classification
from qubo_nn.config import Config
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("type")
parser.add_argument("cmd")
parser.add_argument("cfg_id")
args = parser.parse_args()

cfg = Config().get_cfg(args.cfg_id)

if args.type == 'classify':
    print(args.cmd)
    Classification(cfg).run_experiment()
