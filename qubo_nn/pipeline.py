import gzip
import json
import socket
import random
import pickle
import datetime
import itertools

import torch
import qubovert
import pyxis as px
import numpy as np
import networkx as nx
import qubo_nn.problems
from qubo_nn.nn import Optimizer
from qubo_nn.nn import ReverseOptimizer
from qubo_nn.nn import AutoEncoderOptimizer
from qubo_nn.nn import RNNOptimizer
from qubo_nn.nn import A3Optimizer
from qubo_nn.nn import Resistance1
from qubo_nn.nn import Resistance2
from qubo_nn.nn import QbsolvOptimizer
from qubo_nn.nn import RedAEOptimizer
from qubo_nn.logger import Logger
from qubo_nn.problems import PROBLEM_REGISTRY
from qubo_nn.data import LMDBDataLoader


np.set_printoptions(suppress=True)


class Classification:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks = cfg['problems']['chunks']
        self.n_problems = cfg['problems']['n_problems']
        self.qubo_size = cfg['problems']['qubo_size']
        self.scramble_qubos = cfg['problems']['scramble_qubos']
        self.problems = self._prep_problems()
        self.generalization_cfgs = self.cfg['problems'].get('generalization', {})

    def _prep_problems(self):
        ret = []
        for name in self.cfg['problems']['problems']:
            ret.append((PROBLEM_REGISTRY[name], self.cfg['problems'][name], name))
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
        data = []
        # data = np.zeros(
        #     shape=(len(self.problems) * n_problems, qubo_size, qubo_size),
        #     dtype=np.float32
        # )

        generalized_params = self.cfg['problems'].get('generalization', {})
        generalized_order = 0
        if generalized_params:
            k = list(generalized_params.keys())[0]
            generalized_order = len(generalized_params[k])

        labels = []
        # labels = np.zeros(
        #     shape=(len(self.problems) * n_problems * (generalized_order + 1),),
        #     dtype=np.long
        # )
        for i, (cls, kwargs, name) in enumerate(self.problems):
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )
            generalized_params = self.generalization_cfgs.get(name, [])

            for params in generalized_params:
                args = kwargs.copy()
                args.update(params)
                problems_, qubo_matrices_ = self.gen_qubo_matrices(
                    cls, n_problems, **args
                )

                # Padding
                for k, matrix in enumerate(qubo_matrices_):
                    pad_width = qubo_size - matrix.shape[0]
                    matrix = np.pad(matrix, ((0, pad_width), (0, pad_width)), constant_values=0)
                    qubo_matrices_[k] = matrix
                problems.extend(problems_)
                qubo_matrices.extend(qubo_matrices_)

            all_problems.append(problems)
            if self.scramble_qubos:
                for j in range(len(qubo_matrices)):
                    if random.random() > .5:
                        rand_idx1 = random.randint(0, self.qubo_size - 1)
                        rand_idx2 = random.randint(0, self.qubo_size - 1)
                        val = qubo_matrices[j][[rand_idx2, rand_idx1]]
                        qubo_matrices[j][[rand_idx1, rand_idx2]] = val
            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            # Without normalization of some sort the NN won't learn.
            if not self.cfg["model"]["no_norm"]:
                if self.cfg["model"]["use_norm_divide"]:
                    qubo_matrices /= self.cfg["model"]["norm_multiply"]
                elif self.cfg["model"]["use_norm_multiply_input"]:
                    qubo_matrices /= np.max(np.abs(qubo_matrices))
                    qubo_matrices *= self.cfg["model"]["norm_multiply"]
                elif self.cfg["model"].get("norm_data_abs", False):
                    qubo_matrices /= np.max(np.abs(qubo_matrices))
                elif self.cfg["model"]["norm_data"]:
                    qubo_matrices /= np.max(np.abs(qubo_matrices))
                    qubo_matrices = (qubo_matrices + 1) / 2.
                else:
                    qubo_matrices = (
                        qubo_matrices - np.mean(qubo_matrices)
                    ) / np.std(qubo_matrices)

            # TODO !!
            # if qubo_matrices.shape[0] < n_problems:
            #     data[idx_start:idx_start + qubo_matrices.shape[0], :, :] = qubo_matrices
            # else:
            #     data[idx_start:idx_end, :, :] = qubo_matrices
            data.extend(qubo_matrices)
            # labels[idx_start:idx_end] = i
            labels.extend([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels, all_problems

    def gen_data_gzip_pickle(self):
        data, labels, _ = self._gen_data(self.n_problems)
        fname = 'datasets/%s.pickle.gz' % self.cfg['cfg_id']
        with gzip.open(fname, 'wb+') as f:
            pickle.dump((data, labels), f)

    def gen_data_lmdb(self):
        data, labels, _ = self._gen_data(self.n_problems)
        dirpath = 'datasets/%s/'
        if self.cfg["use_big"]:
            dirpath = '/big/r/ratke/qubo_datasets/%s/'
        db = px.Writer(
            dirpath=dirpath % self.cfg['cfg_id'],
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


class ReverseRegression(Classification):
    def __init__(self, cfg):
        super(ReverseRegression, self).__init__(cfg)

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
                    new_p[0][clause[0][0]][clause[1][0]] -= 2
                    new_p[0][clause[1][0]][clause[0][0]] -= 2
                if not clause[0][1] and not clause[1][1]:
                    new_p[0][clause[0][0]][clause[1][0]] += 1
                    new_p[0][clause[1][0]][clause[0][0]] += 1
                if clause[0][1] and not clause[1][1]:
                    new_p[0][clause[0][0]][clause[1][0]] -= 1
                    new_p[0][clause[1][0]][clause[0][0]] -= 1
                if not clause[0][1] and clause[1][1]:
                    new_p[0][clause[0][0]][clause[1][0]] += 2
                    new_p[0][clause[1][0]][clause[0][0]] += 2
            all_problems[0][i]["clauses"] = list(new_p.flat)

    def gen_apply_m3sat_customization(self, all_problems):
        """Apply a different labelling for M3SAT."""
        for i, p in enumerate(all_problems[0]):
            clauses = p["clauses"]
            new_p = np.zeros(
                shape=(1, self.qubo_size, self.qubo_size, self.qubo_size),
                dtype=np.float32
            )

            for clause in clauses:
                if clause[0][1] and clause[1][1] and clause[2][1]:
                    value = -4
                if clause[0][1] and not clause[1][1] and clause[2][1]:
                    value = -3
                if clause[0][1] and clause[1][1] and not clause[2][1]:
                    value = -2
                if clause[0][1] and not clause[1][1] and not clause[2][1]:
                    value = -1
                if not clause[0][1] and clause[1][1] and clause[2][1]:
                    value = 1
                if not clause[0][1] and not clause[1][1] and clause[2][1]:
                    value = 2
                if not clause[0][1] and not clause[1][1] and not clause[2][1]:
                    value = 3
                if not clause[0][1] and clause[1][1] and not clause[2][1]:
                    value = 4

                for permutation in itertools.permutations(clause, 3):
                    x = permutation[0][0]
                    y = permutation[1][0]
                    z = permutation[2][0]
                    new_p[0][x][y][z] += value
            all_problems[0][i]["clauses"] = list(new_p.flat)

    def gen_data_lmdb(self):
        data, labels, all_problems = self._gen_data(self.n_problems)

        if self.cfg['problems']['problems'] == ["M2SAT"]:
            self.gen_apply_m2sat_customization(all_problems)
            print(data[0])
            print(all_problems[0][0]["clauses"])

        if self.cfg['problems']['problems'] == ["M3SAT"]:
            len_ = len(all_problems[0])
            data = data[:len_]
            labels = labels[:len_]

            self.gen_apply_m3sat_customization(all_problems)
            print(data[0])
            print(all_problems[0][0]["clauses"])

        if self.cfg['problems']['problems'] == ["QA"]:
            print("Simplifying input space..")

            new_data = []

            size = self.cfg["problems"]["QA"]["size"]
            max_full_size = None

            generalized_params = self.cfg['problems'].get('generalization', {}).get('QA', [])
            for m in range(len(generalized_params) + 1):
                if m > 0:
                    size = self.cfg["problems"]["generalization"][m - 1]["size"]
                qubo_size = int(size ** 2)
                for n in range(self.n_problems):
                    d = data[self.n_problems * m + n]
                    d = d[:qubo_size, :qubo_size]
                    tmp_data = []
                    triu_idx = np.triu_indices(size, 1)
                    for i, j in zip(*triu_idx):
                        quadrant = d[i*size:(i+1)*size, j*size:(j+1)*size]
                        x = quadrant[np.triu_indices(size, 1)]
                        tmp_data.extend(x)

                    if m > 0:
                        pad_width = max_full_size - len(tmp_data)
                        tmp_data = np.pad(tmp_data, (0, pad_width))

                    new_data.append(tmp_data)
                if m == 0:
                    max_full_size = len(new_data[0])
            data = np.array(new_data)

            rm_perc = self.cfg["problems"]["QA"].get("remove_percentile_smallest", None)
            if rm_perc is not None:
                for d in data:
                    perc = np.percentile(d, rm_perc)
                    d[np.where(d < perc)] = 0.

            print(data[0])
            print(all_problems[0][0])

        if self.cfg['problems']['problems'] == ["SP"]:
            len_ = len(all_problems[0])
            data = data[:len_]
            labels = labels[:len_]

            print(data[0])
            print(all_problems[0][0])

        if self.cfg['problems']['problems'] == ["QK"]:
            print(data[0])
            print(all_problems[0][0])

            special_norm = self.cfg['problems']['QK'].get('special_norm', False)
            # Assumption: We know where the budgets lie.
            budgets_idx = self.cfg['problems']['QK']['size']
            multiplier = 2 ** (self.qubo_size - budgets_idx)

            print("QK SPEC NORM?", special_norm, budgets_idx, multiplier)

            if special_norm:
                new_data = []
                for k, d in enumerate(data):
                    d_new = d[:budgets_idx, :budgets_idx]
                    y = d[:budgets_idx, budgets_idx].reshape(1, -1)

                    # np.fill_diagonal(d_new, np.diag(d_new) + y * 2 * 6)
                    np.fill_diagonal(d_new, np.diag(d_new) + y * 2 * multiplier)

                    d_new += np.amax([d_new, y.T @ y], axis=0)

                    x = np.concatenate((d_new.flatten(), (-y * 2).flatten(), (y.T @ y).flatten() / 100.))
                    new_data.append(x)
                data = np.array(new_data)
                print(data[0])

        if self.cfg['problems']['problems'] == ["MCQ"]:
            len_ = len(all_problems[0])
            data = data[:len_]
            labels = labels[:len_]

            data -= 2  # NOTE: LOL

            print(data[0])
            print(all_problems[0][0])

        all_problems_flat, output_size = self.flatten_problem_parameters(all_problems)

        prob_type = self.cfg['problems']['problems']
        if prob_type == ["MVC"] or prob_type == ["MC"]:
            for i, prob in enumerate(all_problems_flat):
                len_ = int(len(prob) ** .5)
                pad_width = int(output_size ** .5) - len_
                prob = np.reshape(prob, (len_, len_))
                all_problems_flat[i] = np.pad(
                    prob,
                    ((0, pad_width), (0, pad_width)),
                    'constant',
                    constant_values=0
                ).astype(float).reshape(output_size)
        elif prob_type == ["GC"]:
            for i, prob in enumerate(all_problems_flat):
                len_ = int(len(prob) ** .5)
                pad_width = int(output_size ** .5) - len_
                val = prob[-1]
                prob = np.reshape(prob[:-1], (len_, len_))
                all_problems_flat[i] = np.pad(
                    prob,
                    ((0, pad_width), (0, pad_width)),
                    'constant',
                    constant_values=0
                ).astype(float).reshape(output_size - 1)
                all_problems_flat[i] = np.append(all_problems_flat[i], val)
        else:
            for i, prob in enumerate(all_problems_flat):
                all_problems_flat[i] = np.pad(
                    prob,
                    (0, output_size - len(prob)),
                    'constant',
                    constant_values=(0, 0)
                ).astype(float)

        all_problems_flat = np.array(all_problems_flat, dtype=float)

        print(data.shape, labels.shape, all_problems_flat.shape)

        print("NORM?", not self.cfg["model"]["no_norm"], self.cfg["model"]["norm_multiply"])

        # NOTE: We are using min max normalization here.. Not standardization
        # like with classification.
        if not self.cfg["model"]["no_norm"]:
            print("use norm multiply?", self.cfg["model"]["use_norm_multiply"])
            if self.cfg["model"]["use_norm_divide"]:
                all_problems_flat /= self.cfg["model"]["norm_multiply"]
            elif self.cfg["model"]["use_norm_multiply"]:
                all_problems_flat /= np.max(np.abs(all_problems_flat))
                all_problems_flat *= self.cfg["model"]["norm_multiply"]
                # qa4_diffnorm used 5., qa4_diffnorm_v2 used 10.
            elif self.cfg["model"].get("norm_data_abs", False):
                all_problems_flat /= np.max(np.abs(all_problems_flat))
            else:
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
        part = self.cfg["model"].get("part", None)
        if part:
            lmdb_loader = LMDBDataLoader(self.cfg, reverse=True, part=part)
        else:
            lmdb_loader = LMDBDataLoader(self.cfg, reverse=True)

        dirpath = 'datasets/'
        if self.cfg["use_big"]:
            dirpath = '/big/r/ratke/qubo_datasets/'
        with open(dirpath + '%s/cfg.pickle' % self.cfg['dataset_id'], 'rb') as f:
            output_size = pickle.load(f)

        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = ReverseOptimizer(self.cfg, lmdb_loader, self.logger, output_size)
            # optimizer = RNNOptimizer(self.cfg, lmdb_loader, self.logger, output_size)
            optimizer.train()
            optimizer.save(self.model_fname)
            self.logger.close()

    def eval(self, model_fname):
        part = self.cfg["model"].get("part", None)
        if part:
            lmdb_loader = LMDBDataLoader(self.cfg, reverse=True, part=part)
        else:
            lmdb_loader = LMDBDataLoader(self.cfg, reverse=True)

        with open('datasets/%s/cfg.pickle' % self.cfg['dataset_id'], 'rb') as f:
            output_size = pickle.load(f)

        self.model_fname = self.get_model_fname()
        self.logger = Logger(self.model_fname, self.cfg)
        optimizer = ReverseOptimizer(self.cfg, lmdb_loader, self.logger, output_size)
        optimizer.load(model_fname, output_size)
        self._eval(optimizer)
        self.logger.close()

    def _eval(self, optimizer):
        avg_loss, problem_losses, tot3_fp, tot3_fn, tot3 = optimizer.eval(0, debug=True)
        print(avg_loss, problem_losses)
        print(tot3_fp, tot3_fn, tot3)
        print((tot3_fp + tot3_fn) / tot3)


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


class DefeatClassification(Classification):
    def __init__(self, cfg):
        super(DefeatClassification, self).__init__(cfg)

    def _gen_data(self, n_problems):
        print("Generating data to defeat classification.")

        qubo_size = self.qubo_size

        all_problems = []
        data = []
        labels = []

        generalized_params = self.cfg['problems'].get('generalization', {})
        generalized_order = 0
        if generalized_params:
            k = list(generalized_params.keys())[0]
            generalized_order = len(generalized_params[k])

        for i, (cls, kwargs, name) in enumerate(self.problems):
            print(name)
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )

            generalized_params = self.generalization_cfgs.get(name, [])

            for params in generalized_params:
                args = kwargs.copy()
                args.update(params)
                problems_, qubo_matrices_ = self.gen_qubo_matrices(
                    cls, n_problems, **args
                )

                # Padding
                for k, matrix in enumerate(qubo_matrices_):
                    pad_width = qubo_size - matrix.shape[0]
                    matrix = np.pad(matrix, ((0, pad_width), (0, pad_width)), constant_values=0)
                    qubo_matrices_[k] = matrix
                problems.extend(problems_)
                qubo_matrices.extend(qubo_matrices_)

            do_add_noise = name == "MC" or name == "MVC"
            # do_add_noise = False

            noise_divisor = self.cfg['model']['noise_divisor']

            all_problems.append(problems)
            for j in range(len(qubo_matrices)):
                if self.scramble_qubos:
                    if random.random() > .5:
                        rand_idx1 = random.randint(0, self.qubo_size - 1)
                        rand_idx2 = random.randint(0, self.qubo_size - 1)
                        val = qubo_matrices[j][[rand_idx2, rand_idx1]]
                        qubo_matrices[j][[rand_idx1, rand_idx2]] = val

                if name == "MC":
                    qubo_matrices[j] *= -1
                    # qubo_matrices[j] /= 2.

                if name == "MVC":
                    qubo_matrices[j] /= 8.

                if do_add_noise:
                    noise = np.random.random(size=qubo_matrices[j].shape) / noise_divisor
                    np.fill_diagonal(noise, 0.)
                    qubo_matrices[j] += noise

                # Normalize.
                qubo_matrices[j] /= abs(qubo_matrices[j]).max()

                if name == "MVC" or name == "MC":
                    # qubo_matrices[j][np.where(qubo_matrices[j] > 0.05)] = 0.03
                    qubo_matrices[j][np.where(qubo_matrices[j] > 0.03)] = 0.03
                if name == "GC" or name == "TSP":
                    diag = np.diag(qubo_matrices[j])
                    qubo_matrices[j][np.where(qubo_matrices[j] > 0.03)] = 0.03
                    np.fill_diagonal(qubo_matrices[j], diag / 2)
            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            data.extend(qubo_matrices)
            labels.extend([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels, all_problems


class A3(Classification):
    def __init__(self, cfg):
        super(A3, self).__init__(cfg)

    def solve_qubo2(self, item):
        Q = qubovert.utils.matrix_to_qubo(item.reshape(self.qubo_size, self.qubo_size))
        sol = Q.solve_bruteforce(all_solutions=False)
        sol_ = [0 for _ in range(self.qubo_size)]
        for k, v in sol.items():
            sol_[k] = v
        return sol_

    def _gen_data(self, n_problems):
        print("Generating data for A3.")

        qubo_size = self.qubo_size

        all_problems = []
        data = []
        labels = []

        generalized_params = self.cfg['problems'].get('generalization', {})
        generalized_order = 0
        if generalized_params:
            k = list(generalized_params.keys())[0]
            generalized_order = len(generalized_params[k])

        for i, (cls, kwargs, name) in enumerate(self.problems):
            print(name)
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )

            generalized_params = self.generalization_cfgs.get(name, [])

            for params in generalized_params:
                args = kwargs.copy()
                args.update(params)
                problems_, qubo_matrices_ = self.gen_qubo_matrices(
                    cls, n_problems, **args
                )

                # Padding
                for k, matrix in enumerate(qubo_matrices_):
                    pad_width = qubo_size - matrix.shape[0]
                    matrix = np.pad(matrix, ((0, pad_width), (0, pad_width)), constant_values=0)
                    qubo_matrices_[k] = matrix
                problems.extend(problems_)
                qubo_matrices.extend(qubo_matrices_)

            # do_add_noise = name == "MC" or name == "MVC"

            noise_divisor = self.cfg['model']['noise_divisor']

            all_problems.append(problems)
            for j in range(len(qubo_matrices)):
                if self.scramble_qubos:
                    if random.random() > .5:
                        rand_idx1 = random.randint(0, self.qubo_size - 1)
                        rand_idx2 = random.randint(0, self.qubo_size - 1)
                        val = qubo_matrices[j][[rand_idx2, rand_idx1]]
                        qubo_matrices[j][[rand_idx1, rand_idx2]] = val

                if name == "MC":
                    qubo_matrices[j] *= -1
                    # qubo_matrices[j] /= 2.

                # if name == "MVC":
                #     qubo_matrices[j] /= 8.

                # if do_add_noise:
                #     noise = np.random.random(size=qubo_matrices[j].shape) / noise_divisor
                #     np.fill_diagonal(noise, 0.)
                #     qubo_matrices[j] += noise

                # Normalize.
                # qubo_matrices[j] /= abs(qubo_matrices[j]).max()
                qubo_matrices[j] = qubo_matrices[j] / abs(qubo_matrices[j]).max()

                # if name == "MVC" or name == "MC":
                #     # qubo_matrices[j][np.where(qubo_matrices[j] > 0.05)] = 0.03
                #     qubo_matrices[j][np.where(qubo_matrices[j] > 0.03)] = 0.03
                # if name == "GC" or name == "TSP":
                #     diag = np.diag(qubo_matrices[j])
                #     qubo_matrices[j][np.where(qubo_matrices[j] > 0.03)] = 0.03
                #     np.fill_diagonal(qubo_matrices[j], diag / 2)
            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            data.append(qubo_matrices)  # NOTE: We append, not extend here!!
            labels.append([self.solve_qubo2(q) for q in qubo_matrices])
            # labels.append([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels, all_problems

    def run_experiment(self, n_runs=1):
        print("Running A3.")
        print(torch.get_num_threads())
        lmdb_loader = LMDBDataLoader(self.cfg)
        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = A3Optimizer(self.cfg, lmdb_loader, self.logger)
            optimizer.train()
            optimizer.save(self.model_fname)
            self._eval(optimizer)
            self.logger.close()

    def _eval(self, optimizer):
        # TODO
        pass


class R1(Classification):
    def __init__(self, cfg):
        super(R1, self).__init__(cfg)

    def _gen_data(self, n_problems):
        print("Generating data for R1.")

        qubo_size = self.qubo_size

        all_problems = []
        data = []
        labels = []

        generalized_params = self.cfg['problems'].get('generalization', {})
        generalized_order = 0
        if generalized_params:
            k = list(generalized_params.keys())[0]
            generalized_order = len(generalized_params[k])

        for i, (cls, kwargs, name) in enumerate(self.problems):
            print(name)
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )

            generalized_params = self.generalization_cfgs.get(name, [])

            for params in generalized_params:
                args = kwargs.copy()
                args.update(params)
                problems_, qubo_matrices_ = self.gen_qubo_matrices(
                    cls, n_problems, **args
                )

                # Padding
                for k, matrix in enumerate(qubo_matrices_):
                    pad_width = qubo_size - matrix.shape[0]
                    matrix = np.pad(matrix, ((0, pad_width), (0, pad_width)), constant_values=0)
                    qubo_matrices_[k] = matrix
                problems.extend(problems_)
                qubo_matrices.extend(qubo_matrices_)

            all_problems.append(problems)
            for j in range(len(qubo_matrices)):
                if self.scramble_qubos:
                    if random.random() > .5:
                        rand_idx1 = random.randint(0, self.qubo_size - 1)
                        rand_idx2 = random.randint(0, self.qubo_size - 1)
                        val = qubo_matrices[j][[rand_idx2, rand_idx1]]
                        qubo_matrices[j][[rand_idx1, rand_idx2]] = val

                if name == "MC":
                    qubo_matrices[j] *= -1

                # Normalize.
                qubo_matrices[j] /= abs(qubo_matrices[j]).max()

            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            data.extend(qubo_matrices)
            labels.extend([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels, all_problems

    def run_experiment(self, n_runs=1):
        print("Running R1.")
        lmdb_loader = LMDBDataLoader(self.cfg)
        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = Resistance1(self.cfg, lmdb_loader, self.logger)
            optimizer.train()
            optimizer.save(self.model_fname)
            self._eval(optimizer)
            self.logger.close()

    def _eval(self, optimizer):
        # TODO
        pass


class R2(Classification):
    def __init__(self, cfg):
        super(R2, self).__init__(cfg)

    def _gen_data(self, n_problems):
        print("Generating data for R2.")

        qubo_size = self.qubo_size

        all_problems = []
        data = []
        labels = []

        generalized_params = self.cfg['problems'].get('generalization', {})
        generalized_order = 0
        if generalized_params:
            k = list(generalized_params.keys())[0]
            generalized_order = len(generalized_params[k])

        for i, (cls, kwargs, name) in enumerate(self.problems):
            print(name)
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )

            generalized_params = self.generalization_cfgs.get(name, [])

            for params in generalized_params:
                args = kwargs.copy()
                args.update(params)
                problems_, qubo_matrices_ = self.gen_qubo_matrices(
                    cls, n_problems, **args
                )

                # Padding
                for k, matrix in enumerate(qubo_matrices_):
                    pad_width = qubo_size - matrix.shape[0]
                    matrix = np.pad(matrix, ((0, pad_width), (0, pad_width)), constant_values=0)
                    qubo_matrices_[k] = matrix
                problems.extend(problems_)
                qubo_matrices.extend(qubo_matrices_)

            all_problems.append(problems)
            for j in range(len(qubo_matrices)):
                if self.scramble_qubos:
                    if random.random() > .5:
                        rand_idx1 = random.randint(0, self.qubo_size - 1)
                        rand_idx2 = random.randint(0, self.qubo_size - 1)
                        val = qubo_matrices[j][[rand_idx2, rand_idx1]]
                        qubo_matrices[j][[rand_idx1, rand_idx2]] = val

                if name == "MC":
                    qubo_matrices[j] *= -1

                # Normalize.
                qubo_matrices[j] /= abs(qubo_matrices[j]).max()

            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            data.extend(qubo_matrices)
            labels.extend([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels, all_problems

    def run_experiment(self, n_runs=1):
        print("Running R2.")
        lmdb_loader = LMDBDataLoader(self.cfg)
        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = Resistance2(self.cfg, lmdb_loader, self.logger)
            optimizer.train()
            optimizer.save(self.model_fname)
            self._eval(optimizer)
            self.logger.close()

    def _eval(self, optimizer):
        # TODO
        pass


class QbsolvRegression(Classification):
    def __init__(self, cfg):
        super(QbsolvRegression, self).__init__(cfg)
        self.qubo_size = self.cfg["problems"]["qubo_size"]

    def gen_qubo_matrices(self, cls, n_problems, **kwargs):
        problems = cls.gen_problems(self.cfg, n_problems, **kwargs)
        qubo_matrices = []
        for i, problem in enumerate(problems):
            if i % 100000 == 0:
                print(i)
            qubo_matrices.append(cls(self.cfg, **problem).gen_qubo_matrix())
            del problem
        return qubo_matrices

    def _gen_data(self, n_problems):
        qubo_size = self.qubo_size
        data = []

        labels = []
        for i, (cls, kwargs, name) in enumerate(self.problems):
            cls = qubo_nn.problems.max_cut.MaxCutMemoryEfficient
            qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )

            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            data.extend(qubo_matrices)
            labels.extend([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.dtype('b'))
        labels = np.array(labels, dtype=np.int32)
        return data, labels, []

    def solve_qubo(self, qubo):
        qubo = qubo.copy().astype(np.dtype('i4'))
        Q = qubovert.utils.matrix_to_qubo(qubo)
        sol = Q.solve_bruteforce(all_solutions=False)
        sol_ = [0 for _ in range(self.qubo_size)]
        for k, v in sol.items():
            sol_[k] = v
        # return torch.FloatTensor(sol_)
        return sol_

    def gen_data_lmdb(self):
        data, _, _ = self._gen_data(self.n_problems)

        labels = []

        for i, qubo in enumerate(data):
            if i % 100000 == 0:
                print(i)
            labels.append(self.solve_qubo(qubo))
        labels = np.array(labels)

        print(data.shape, labels.shape)

        print("NORM?", not self.cfg["model"]["no_norm"], self.cfg["model"]["norm_multiply"])

        db = px.Writer(
            dirpath='datasets/%s/' % self.cfg['cfg_id'],
            map_size_limit=60000,
            ram_gb_limit=60
        )
        db.put_samples('input', data, 'target', labels)
        db.close()

    def run_experiment(self, n_runs=1):
        lmdb_loader = LMDBDataLoader(self.cfg)

        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = QbsolvOptimizer(self.cfg, lmdb_loader, self.logger)
            optimizer.train()
            optimizer.save(self.model_fname)
            self.logger.close()


class RedAERegression(Classification):
    def __init__(self, cfg):
        super(RedAERegression, self).__init__(cfg)
        self.qubo_size = self.cfg["problems"]["qubo_size"]

    def gen_qubo_matrices(self, cls, n_problems, **kwargs):
        problems = cls.gen_problems(self.cfg, n_problems, **kwargs)
        qubo_matrices = []
        for i, problem in enumerate(problems):
            if i % 100000 == 0:
                print(i)
            qubo_matrices.append(cls(self.cfg, **problem).gen_qubo_matrix())
            del problem
        return qubo_matrices

    def _gen_data(self, n_problems):
        qubo_size = self.qubo_size
        data = []

        labels = []
        for i, (cls, kwargs, name) in enumerate(self.problems):
            cls = qubo_nn.problems.max_cut.MaxCutMemoryEfficient
            qubo_matrices = self.gen_qubo_matrices(
                cls, n_problems, **kwargs
            )

            qubo_matrices = np.array(qubo_matrices)
            print(cls, qubo_matrices.shape)

            data.extend(qubo_matrices)
            labels.extend([i for _ in range(len(qubo_matrices))])

        print("TOTAL DATA LEN", len(data))
        data = np.array(data, dtype=np.dtype('b'))
        labels = np.array(labels, dtype=np.int32)
        return data, labels, []

    def gen_data_lmdb(self):
        data, labels, _ = self._gen_data(self.n_problems)
        print(data.shape, labels.shape)

        print("NORM?", not self.cfg["model"]["no_norm"], self.cfg["model"]["norm_multiply"])

        db = px.Writer(
            dirpath='datasets/%s/' % self.cfg['cfg_id'],
            map_size_limit=60000,
            ram_gb_limit=60
        )
        db.put_samples('input', data, 'target', labels)
        db.close()

    def run_experiment(self, n_runs=1):
        lmdb_loader = LMDBDataLoader(self.cfg)

        for _ in range(n_runs):
            self.model_fname = self.get_model_fname()
            self.logger = Logger(self.model_fname, self.cfg)
            self.logger.log_config()
            optimizer = RedAEOptimizer(self.cfg, lmdb_loader, self.logger)
            optimizer.train()
            optimizer.save(self.model_fname)
            self.logger.close()
