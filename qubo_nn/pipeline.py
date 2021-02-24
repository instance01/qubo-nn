import gzip
import json
import socket
import random
import pickle
import datetime

import pyxis as px
import numpy as np
from qubo_nn.nn import Optimizer
from qubo_nn.logger import Logger
from qubo_nn.problems import PROBLEM_REGISTRY
from qubo_nn.data import ChunkedDataLoader
from qubo_nn.data import LMDBDataLoader


class Classification:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks = cfg['problems']['chunks']
        self.n_problems = cfg['problems']['n_problems']
        self.qubo_size = cfg['problems']['qubo_size']
        self.problems = self._prep_problems()
        self.model_fname = self.get_model_fname()
        self.logger = Logger(self.model_fname, cfg)

    def _prep_problems(self):
        ret = []
        for name in self.cfg['problems']['problems']:
            ret.append((PROBLEM_REGISTRY[name], self.cfg['problems'][name]))
        return ret

    def gen_qubo_matrices(self, cls, n_problems, **kwargs):
        problems = cls.gen_problems(n_problems, **kwargs)
        qubo_matrices = [
            cls(**problem).gen_qubo_matrix()
            for problem in problems
        ]
        return qubo_matrices

    def _gen_data(self, n_problems):
        qubo_size = self.qubo_size
        data = np.zeros(
            shape=(len(self.problems) * n_problems, qubo_size, qubo_size),
            dtype=np.float32
        )
        labels = np.zeros(
            shape=(len(self.problems) * n_problems,),
            dtype=np.long
        )
        for i, (cls, args) in enumerate(self.problems):
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **args
            )
            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            if self.cfg["model"]["norm_data"]:
                qubo_matrices /= np.max(np.abs(qubo_matrices))
                qubo_matrices = (qubo_matrices + 1) / 2.
            else:
                # TODO DUBIOUS!!!
                # This should be an option. But for now without it the neural
                # network won't learn.
                qubo_matrices = (
                    qubo_matrices - np.mean(qubo_matrices)
                ) / np.std(qubo_matrices)

            data[idx_start:idx_end, :, :] = qubo_matrices
            labels[idx_start:idx_end] = i

        return data, labels

    def gen_data_gzip_pickle(self):
        data, labels = self._gen_data(self.n_problems)
        with gzip.open('datasets/%s.pickle.gz' % self.cfg['cfg_id'], 'wb+') as f:
            pickle.dump((data, labels), f)

    def gen_data_lmdb(self):
        data, labels = self._gen_data(self.n_problems)
        db = px.Writer(dirpath='datasets/%s/' % self.cfg['cfg_id'], map_size_limit=60000, ram_gb_limit=60)
        db.put_samples('input', data, 'target', labels)
        db.close()

    def gen_data_chunks(self):
        n_problems = self.n_problems
        for chunk in range(self.chunks):
            n_problems = self.n_problems // self.chunks
            data, labels = self._gen_data(n_problems)

            with gzip.open('datasets/%s.%d.pickle.gz' % (self.cfg['cfg_id'], chunk), 'wb+') as f:
                pickle.dump((data, labels), f)

    def run_experiment(self):
        self.logger.log_config()
        lmdb_loader = LMDBDataLoader(self.cfg)
        optimizer = Optimizer(self.cfg, lmdb_loader, self.logger)
        optimizer.train()
        optimizer.save(self.model_fname)
        self._eval(optimizer)

    def eval(self, model_fname):
        lmdb_loader = LMDBDataLoader(self.cfg)
        optimizer = Optimizer(self.cfg, lmdb_loader, self.logger)
        optimizer.load(model_fname)
        self._eval(optimizer)

    def _eval(self, optimizer):
        misclassifications, _, _, mc_table = optimizer.eval()
        mc_prob = {
            self.cfg['problems']['problems'][int(k)]: v
            for k, v in misclassifications.items()
        }
        print(json.dumps(mc_prob, indent=4))
        self.logger.log_confusion_matrix(mc_table)

    def get_model_fname(self):
        rand_str = str(int(random.random() * 10e6))
        model_fname = "-".join([
            datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S"),
            rand_str,
            socket.gethostname(),
            self.cfg['cfg_id']
        ])
        return model_fname

    def close(self):
        self.logger.close()
