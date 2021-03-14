from qubo_nn.pipeline import Classification
from qubo_nn.pipeline import ReverseClassification
from qubo_nn.config import Config
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", nargs=None, help="Type (classify)", default='classify')
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gendata", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("-c", "--cfg_id", nargs=None, help="cfg_id")
    parser.add_argument("-m", "--model", nargs="?")
    parser.add_argument("-n", "--nruns", nargs="?", type=int, default=3)
    args = parser.parse_args()

    # TODO: Some optionals are 'too optional' - i.e. I use them but they're optional

    cfg = Config().get_cfg(args.cfg_id)

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
            c = ReverseClassification(cfg)
            c.gen_data_lmdb()
        elif args.train:
            c = ReverseClassification(cfg)
            c.run_experiment(args.nruns)
