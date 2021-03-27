import gzip
import json
import socket
import random
import pickle
import datetime

import pyxis as px
import numpy as np
import networkx as nx
from qubo_nn.nn import Optimizer
from qubo_nn.nn import ReverseOptimizer
from qubo_nn.nn import AutoEncoderOptimizer
from qubo_nn.logger import Logger
from qubo_nn.problems import PROBLEM_REGISTRY
from qubo_nn.data import LMDBDataLoader


class Classification:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks = cfg['problems']['chunks']
        self.n_problems = cfg['problems']['n_problems']
        self.qubo_size = cfg['problems']['qubo_size']
        self.scramble_qubos = cfg['problems']['scramble_qubos']
        self.problems = self._prep_problems()

    def _prep_problems(self):
        ret = []
        for name in self.cfg['problems']['problems']:
            ret.append((PROBLEM_REGISTRY[name], self.cfg['problems'][name]))
        return ret

    def gen_qubo_matrices(self, cls, n_problems, **kwargs):
        problems = cls.gen_problems(self.cfg, n_problems, **kwargs)
        qubo_matrices = [
            cls(self.cfg, **problem).gen_qubo_matrix()
            for problem in problems
        ]
        return problems, qubo_matrices

    def _gen_data(self, n_problems):
        qubo_size = self.qubo_size
        all_problems = []
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
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **args
            )
            all_problems.append(problems)
            if self.scramble_qubos:
                for j in range(n_problems):
                    if random.random() > .5:
                        rand_idx1 = random.randint(0, self.qubo_size - 1)
                        rand_idx2 = random.randint(0, self.qubo_size - 1)
                        val = qubo_matrices[j][[rand_idx2, rand_idx1]]
                        qubo_matrices[j][[rand_idx1, rand_idx2]] = val
            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            # Without normalization of some sort the NN won't learn.
            if not self.cfg["model"]["no_norm"]:
                if self.cfg["model"]["norm_data"]:
                    qubo_matrices /= np.max(np.abs(qubo_matrices))
                    qubo_matrices = (qubo_matrices + 1) / 2.
                else:
                    qubo_matrices = (
                        qubo_matrices - np.mean(qubo_matrices)
                    ) / np.std(qubo_matrices)

            data[idx_start:idx_end, :, :] = qubo_matrices
            labels[idx_start:idx_end] = i

        return data, labels, all_problems

    def gen_data_gzip_pickle(self):
        data, labels, _ = self._gen_data(self.n_problems)
        fname = 'datasets/%s.pickle.gz' % self.cfg['cfg_id']
        with gzip.open(fname, 'wb+') as f:
            pickle.dump((data, labels), f)

    def gen_data_lmdb(self):
        data, labels, _ = self._gen_data(self.n_problems)
        db = px.Writer(
            dirpath='datasets/%s/' % self.cfg['cfg_id'],
            map_size_limit=60000,
            ram_gb_limit=60
        )
        db.put_samples('input', data, 'target', labels)
        db.close()

    def gen_data_chunks(self):
        n_problems = self.n_problems
        for chunk in range(self.chunks):
            n_problems = self.n_problems // self.chunks
            data, labels, _ = self._gen_data(n_problems)

            fname = 'datasets/%s.%d.pickle.gz' % (self.cfg['cfg_id'], chunk)
            with gzip.open(fname, 'wb+') as f:
                pickle.dump((data, labels), f)

    def run_experiment(self, n_runs=1):
        lmdb_loader = LMDBDataLoader(self.cfg)
        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = Optimizer(self.cfg, lmdb_loader, self.logger)
            optimizer.train()
            optimizer.save(self.model_fname)
            self._eval(optimizer)
            self.logger.close()

    def eval(self, model_fname):
        lmdb_loader = LMDBDataLoader(self.cfg)
        self.model_fname = self.get_model_fname()
        self.logger = Logger(self.model_fname, self.cfg)
        optimizer = Optimizer(self.cfg, lmdb_loader, self.logger)
        optimizer.load(model_fname)
        self._eval(optimizer)
        self.logger.close()

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


class ReverseClassification(Classification):
    def __init__(self, cfg):
        super(ReverseClassification, self).__init__(cfg)

    def flatten_problem_parameters(self, all_problems):

        # TODO: Absolutely horrendous.

        gen_edges = self.cfg["model"]["gen_edges"]

        output_size = 0
        result = []
        for problem_class_params in all_problems:
            for problem in problem_class_params:
                problem = list(problem.values())
                curr_problem_result = []
                for part in problem:
                    if isinstance(part, nx.classes.graph.Graph):
                        if gen_edges:
                            A = np.asarray(sorted(part.edges))
                        else:
                            A = nx.to_numpy_matrix(part)
                        curr_problem_result.extend(list(A.flat))
                    elif isinstance(part, int) or isinstance(part, float):
                        curr_problem_result.append(part)
                    elif isinstance(part[0], list) or isinstance(part[0], tuple) or isinstance(part[0], np.ndarray):
                        if isinstance(part[0][0], tuple) or isinstance(part[0][0], list):
                            flat_list = [item for sublist in part for subsublist in sublist for item in subsublist]
                        else:
                            flat_list = [item for sublist in part for item in sublist]
                        flat_list = [float(x) for x in flat_list]
                        curr_problem_result.extend(flat_list)
                    else:
                        curr_problem_result.extend(part)
                result.append(curr_problem_result)
                output_size = max(output_size, len(curr_problem_result))
        return result, output_size

    def gen_apply_m2sat_customization(self, all_problems):
        """Apply a different labelling for M2SAT.

        Specifically, given the QUBO size is 64, we end up with a 64x64 label
        where all fields that have 1 denote a (T,T) clause, all fields that
        have (F,F) have 2, and so on (see code below).

        The original idea was to have the label consist of 4 of those 64x64
        matrices, but this ends up being very tough to train, with a ~16k
        dimensional output..

        Note that while I call this label, this is simply the NN output we
        train against.
        """
        for i, p in enumerate(all_problems[0]):
            clauses = p["clauses"]
            new_p = np.zeros(
                # shape=(4, self.qubo_size, self.qubo_size),
                shape=(1, self.qubo_size, self.qubo_size),
                dtype=np.float32
            )
            for clause in clauses:
                if clause[0][1] and clause[1][1]:
                    new_p[0][clause[0][0]][clause[1][0]] += 1
                    new_p[0][clause[1][0]][clause[0][0]] += 1
                if not clause[0][1] and not clause[1][1]:
                    # new_p[1][clause[0][0]][clause[1][0]] = 1
                    # new_p[1][clause[1][0]][clause[0][0]] = 1
                    new_p[0][clause[0][0]][clause[1][0]] += 2
                    new_p[0][clause[1][0]][clause[0][0]] += 2
                if clause[0][1] and not clause[1][1]:
                    # new_p[2][clause[0][0]][clause[1][0]] = 1
                    # new_p[2][clause[1][0]][clause[0][0]] = 1
                    new_p[0][clause[0][0]][clause[1][0]] += 4
                    new_p[0][clause[1][0]][clause[0][0]] += 4
                if not clause[0][1] and clause[1][1]:
                    # new_p[3][clause[0][0]][clause[1][0]] = 1
                    # new_p[3][clause[1][0]][clause[0][0]] = 1
                    new_p[0][clause[0][0]][clause[1][0]] += 8
                    new_p[0][clause[1][0]][clause[0][0]] += 8
            all_problems[0][i]["clauses"] = list(new_p.flat)

    def gen_data_lmdb(self):
        data, labels, all_problems = self._gen_data(self.n_problems)

        if self.cfg['problems']['problems'] == ["M2SAT"]:
            self.gen_apply_m2sat_customization(all_problems)
            print(data[0])
            print(all_problems[0][0]["clauses"])

        all_problems_flat, output_size = self.flatten_problem_parameters(all_problems)

        for i, prob in enumerate(all_problems_flat):
            all_problems_flat[i] = np.pad(
                prob,
                (0, output_size - len(prob)),
                'constant',
                constant_values=(0, 0)
            ).astype(float)

        all_problems_flat = np.array(all_problems_flat, dtype=float)

        print(data.shape, labels.shape, all_problems_flat.shape)

        # NOTE: We are using min max normalization here.. Not standardization
        # like with classification.
        if not self.cfg["model"]["no_norm"]:
            all_problems_flat /= np.max(np.abs(all_problems_flat))
            all_problems_flat = (all_problems_flat + 1) / 2.

        # all_problems_flat = (
        #     all_problems_flat - np.mean(all_problems_flat)
        # ) / np.std(all_problems_flat)

        db = px.Writer(
            dirpath='datasets/%s/' % self.cfg['cfg_id'],
            map_size_limit=60000,
            ram_gb_limit=60
        )
        db.put_samples('input', data, 'target', labels, 'prob', all_problems_flat)
        db.close()

        with open('datasets/%s/cfg.pickle' % self.cfg['cfg_id'], 'wb+') as f:
            pickle.dump(output_size, f)

    def run_experiment(self, n_runs=1):
        part = self.cfg["model"]["part"]
        if part:
            lmdb_loader = LMDBDataLoader(self.cfg, reverse=True, part=part)
        else:
            lmdb_loader = LMDBDataLoader(self.cfg, reverse=True)

        with open('datasets/%s/cfg.pickle' % self.cfg['dataset_id'], 'rb') as f:
            output_size = pickle.load(f)

        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = ReverseOptimizer(self.cfg, lmdb_loader, self.logger, output_size)
            optimizer.train()
            optimizer.save(self.model_fname)
            self.logger.close()


class AutoEncoder(Classification):
    def auto_encoder_prototype(self):
        # TODO Refactor this into its own class.

        # Prepare data.
        lmdb_loader = LMDBDataLoader(self.cfg)

        # TODO Support n_runs later.
        # for _ in range(n_runs):

        self.model_fname = self.get_model_fname()
        self.logger = Logger(self.model_fname, self.cfg)
        self.logger.log_config()

        optimizer = AutoEncoderOptimizer(self.cfg, lmdb_loader, self.logger)
        optimizer.train()
        optimizer.save(self.model_fname)

        self.logger.close()
