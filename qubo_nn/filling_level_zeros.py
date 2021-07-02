import pyxis as px
import numpy as np

from qubo_nn.config import Config
from qubo_nn.data import LMDBDataLoader


problems_short = ["np", "mc", "mvc", "sp", "m2sat", "spp", "gc", "qa", "qk", "m3sat", "tsp", "gi", "sgi", "mcq"]
QUBO_SIZE = 64


for problem in problems_short:
    cfg = Config().get_cfg('red_%s_1' % problem)
    cfg["use_big"] = False
    lmdb_loader = LMDBDataLoader(cfg)
    loader = lmdb_loader.train_data_loader
    data = list(loader)
    total = 0.
    zeros = 0.
    for batch in data:
        batch = batch[0]
        batch = batch.reshape(100 * 4096)
        total += 100 * 4096
        zeros += np.where(batch == 0.)[0].shape[0]
    # print(total)
    # print(zeros)
    print(problem, zeros / total)
