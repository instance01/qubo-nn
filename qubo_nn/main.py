from qubo_nn.pipeline import Classification
from qubo_nn.pipeline import DefeatClassification
from qubo_nn.pipeline import ReverseRegression
from qubo_nn.pipeline import A3
from qubo_nn.pipeline import R1
from qubo_nn.pipeline import R2
from qubo_nn.pipeline import QbsolvRegression
from qubo_nn.config import Config
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", nargs=None, help="Type (classify, reverse)", default='classify')
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gendata", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--big", action="store_true")
    parser.add_argument("-c", "--cfg_id", nargs=None, help="cfg_id")
    parser.add_argument("-m", "--model", nargs="?")
    parser.add_argument("-n", "--nruns", nargs="?", type=int, default=3)
    args = parser.parse_args()

    # TODO: Some optionals are 'too optional' - i.e. I use them but they're optional

    cfg = Config().get_cfg(args.cfg_id)
    cfg["use_big"] = args.big

    if args.type == 'classify':
        if args.gendata:
            c = Classification(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = Classification(cfg)
            c.run_experiment(args.nruns)
        elif args.eval:
            c = Classification(cfg)
            c.eval(args.model)
    elif args.type == 'autoencode':
        if args.train:
            c = Classification(cfg)
            c.auto_encoder_prototype()
    elif args.type == 'reverse':
        if args.gendata:
            c = ReverseRegression(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = ReverseRegression(cfg)
            c.run_experiment(args.nruns)
        elif args.eval:
            c = ReverseRegression(cfg)
            c.eval(args.model)
    elif args.type == 'defeat':
        if args.gendata:
            c = DefeatClassification(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = DefeatClassification(cfg)
            c.run_experiment(args.nruns)
    elif args.type == 'a3':
        if args.gendata:
            c = A3(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = A3(cfg)
            c.run_experiment(args.nruns)
    elif args.type == 'r1':
        if args.gendata:
            c = R1(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = R1(cfg)
            c.run_experiment(args.nruns)
    elif args.type == 'r2':
        if args.gendata:
            c = R2(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = R2(cfg)
            c.run_experiment(args.nruns)
    elif args.type == 'qbsolv':
        if args.gendata:
            c = QbsolvRegression(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = QbsolvRegression(cfg)
            c.run_experiment(args.nruns)
