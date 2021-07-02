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
    is_diag_same = True
    for batch in data:
        batch = batch[0]
        last_diag = None
        for qubo in batch:
            if last_diag is not None and not np.allclose(np.diag(qubo), last_diag):
                is_diag_same = False
            last_diag = np.diag(qubo)
    # print(total)
    # print(zeros)
    print(problem, is_diag_same)
