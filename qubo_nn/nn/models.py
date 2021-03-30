import sys
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dwave_qbsolv import QBSolv


class FCNet(nn.Module):
    def __init__(self, cfg):
        super(FCNet, self).__init__()
        input_size = cfg['problems']['qubo_size'] ** 2

        activation_type = cfg['model']['activation']
        if activation_type == "ReLU":
            activation_cls = nn.ReLU
        elif activation_type == "ELU":
            activation_cls = nn.ELU
        elif activation_type == "LeakyReLU":
            activation_cls = nn.LeakyReLU

        net = []
        last_fc_size = input_size
        for size in cfg['model']['fc_sizes']:
            net.append(nn.Linear(last_fc_size, size))
            net.append(activation_cls())
            last_fc_size = size

        self.fc_net = nn.Sequential(*net)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        return F.log_softmax(x, dim=1)


# TODO This is hardcoded for now, just a prototype.
class AutoEncoderFCNet(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderFCNet, self).__init__()
        self.input_size = cfg['problems']['qubo_size'] ** 2

        self.encoder = nn.Linear(self.input_size, int(self.input_size / 2))
        self.decoder = nn.Linear(int(self.input_size / 2), self.input_size)
        net = [
            self.encoder,
            nn.LeakyReLU(),
            self.decoder,
            nn.LeakyReLU()
        ]
        self.fc_net = nn.Sequential(*net)

    def predict_encode(self, x):
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        return nn.LeakyReLU()(x)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        size = int(self.input_size ** .5)
        return x.reshape(-1, size, size)


class ReverseFCNet(nn.Module):
    def __init__(self, cfg, output_size):
        super(ReverseFCNet, self).__init__()
        input_size = cfg['problems']['qubo_size'] ** 2

        activation_type = cfg['model']['activation']
        if activation_type == "ReLU":
            activation_cls = nn.ReLU
        elif activation_type == "ELU":
            activation_cls = nn.ELU
        elif activation_type == "LeakyReLU":
            activation_cls = nn.LeakyReLU

        fc_sizes = cfg['model']['fc_sizes'] + [output_size]

        net = []
        last_fc_size = input_size
        for size in fc_sizes:
            net.append(nn.Linear(last_fc_size, size))
            net.append(activation_cls())
            last_fc_size = size

        net.pop(-1)
        self.fc_net = nn.Sequential(*net)
        print(self.fc_net)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc_net(x)


class CNNNet(nn.Module):
    pass  # TODO


class RNNNet(nn.Module):
    pass  # TODO


class Optimizer:
    def __init__(self, cfg, lmdb_loader, logger):
        self.cfg = cfg
        self.lmdb_loader = lmdb_loader
        self.logger = logger

        # Load cfg variables.
        lr = cfg['model']['lr']
        sgd_momentum = cfg['model']['optimizer_sgd_momentum']
        self.batch_size = cfg['model']['batch_size']
        self.n_epochs = cfg['model']['n_epochs']
        self.train_eval_split = cfg['model']['train_eval_split']

        # Set it all up.
        self.net = FCNet(cfg)
        weights = cfg['model']['class_weights']
        if weights:
            weights = torch.FloatTensor(weights)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=lr, momentum=sgd_momentum
        )

    def train(self):
        self.net.train()
        data_len = len(self.lmdb_loader.train_data_loader)
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.lmdb_loader.train_data_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)

                try:
                    loss = self.criterion(outputs, labels)
                except IndexError:
                    print(
                        "Size of last layer should equal the number of "
                        "problems you have"
                    )

                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 100 == 0:
                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.3f' % (epoch + 1, i, avg_loss)
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                if i % 1000 == 0:
                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, data_len * epoch + i)

            self.net.eval()
            data = {}
            misclassifications, test_loss, tot_mc, _ = self.eval(False)
            mc_prob = {
                prob: misclassifications.get(i, 0)
                for i, prob in enumerate(self.cfg['problems']['problems'])
            }
            data['loss_eval'] = test_loss
            data['Problem_Misclassifications'] = {}
            for k, v in mc_prob.items():
                data['Problem_Misclassifications'][k] = v
            data['Total_Misclassifications'] = tot_mc
            self.logger.log_eval(data, epoch)
            self.net.train()
        print('')

    def eval(self, do_print=True):
        misclassifications = collections.defaultdict(int)
        n_classes = len(self.cfg['problems']['problems'])
        mc_table = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)
        n_class_labels = np.zeros(shape=(n_classes,), dtype=np.float32)

        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.lmdb_loader.test_data_loader):
            inputs, labels = data
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            prediction = torch.argmax(outputs)
            if prediction != labels:
                misclassifications[labels.item()] += 1
            mc_table[labels.item()][prediction.item()] += 1
            n_class_labels[labels.item()] += 1

            total_loss += loss.item()
            if do_print and i % 1000 == 0:
                msg = '[%d] loss: %.3f' % (i, total_loss / (i + 1))
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()

        data_len = len(self.lmdb_loader.test_data_loader)
        mc_table /= n_class_labels[:, np.newaxis]
        tot_mc = float(sum(misclassifications.values())) / data_len

        if do_print:
            print('\nMisclassification rate', tot_mc)

        return misclassifications, total_loss / data_len, tot_mc, mc_table

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname):
        self.net = FCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))


# TODO This is just a prototype. Lots of copy&paste from Optimizer - refactor
# at some point.
class AutoEncoderOptimizer:
    def __init__(self, cfg, lmdb_loader, logger):
        self.cfg = cfg
        self.lmdb_loader = lmdb_loader
        self.logger = logger

        # Load cfg variables.
        lr = cfg['model']['lr']
        sgd_momentum = cfg['model']['optimizer_sgd_momentum']
        self.batch_size = cfg['model']['batch_size']
        self.n_epochs = cfg['model']['n_epochs']
        self.train_eval_split = cfg['model']['train_eval_split']

        # Set it all up.
        self.net = AutoEncoderFCNet(cfg)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=lr, momentum=sgd_momentum
        )

    def solve_qubo(self, batch):
        # TODO Don't solve QUBOs on the fly.. This needs fixing.
        ret_batch = []
        qb = QBSolv()
        for item in batch:
            Q = {}
            # We can assume a quadratic matrix.
            for i in range(item.shape[0]):
                for j in range(item.shape[1]):
                    if item[i][j] != 0:
                        Q[(i, j)] = item[i][j]
            response = qb.sample_qubo(Q, num_repeats=100)
            ret = [0] * 64  # TODO Hardcoded
            for k, v in response.samples()[0].items():
                ret[k] = v
            ret_batch.append(ret)
        return torch.FloatTensor(ret_batch)

    def train(self):
        self.net.train()
        data_len = len(self.lmdb_loader.train_data_loader)
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.lmdb_loader.train_data_loader):
                inputs, labels = data
                solutions_inputs = self.solve_qubo(inputs)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                solutions_outputs = self.solve_qubo(outputs)

                try:
                    # TODO: Mar 8, 10:20 , qbsolv breaks autograd. Lets do a normal AE first.
                    # loss = self.criterion(solutions_outputs, solutions_inputs)
                    # loss.requires_grad = True
                    loss = self.criterion(inputs, outputs)
                except IndexError:
                    print(
                        "Size of last layer should equal the number of "
                        "problems you have"
                    )

                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 1 == 0:
                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.3f' % (epoch + 1, i, avg_loss)
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                if i % 1000 == 0:
                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, data_len * epoch + i)
        print('')

    def eval(self, do_print=True):
        # TODO
        return {}, 0, {}, {}

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname):
        self.net = AutoEncoderFCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))


class ReverseOptimizer(Optimizer):
    def __init__(self, cfg, lmdb_loader, logger, output_size):
        self.cfg = cfg
        self.lmdb_loader = lmdb_loader
        self.logger = logger

        # Load cfg variables.
        lr = cfg['model']['lr']
        sgd_momentum = cfg['model']['optimizer_sgd_momentum']
        self.batch_size = cfg['model']['batch_size']
        self.n_epochs = cfg['model']['n_epochs']
        self.train_eval_split = cfg['model']['train_eval_split']

        # Set it all up.
        self.net = ReverseFCNet(cfg, output_size)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        if cfg['model']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=lr, momentum=sgd_momentum
            )
        elif cfg['model']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=lr
            )

    def train(self):
        save_fpfn = self.cfg['problems']['problems'] == ["M2SAT"] or \
            self.cfg['problems']['problems'] == ["M3SAT"]

        self.net.train()
        data_len = len(self.lmdb_loader.train_data_loader)
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.lmdb_loader.train_data_loader):
                inputs, labels, problem = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)

                loss = self.criterion(outputs, problem.float())
                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 100 == 0:
                    X = problem.float().tolist()[-1][:200]
                    Y = outputs.detach().tolist()[-1][:200]
                    print('')
                    for m in range(200):
                        if X[m] > 0 or X[m] < 0:
                            print(X[m], round(Y[m], 3))
                    # print([round(x, 2) for x in outputs.detach().tolist()[-1][:100]])
                    # print(problem.float().tolist()[-1][:100])
                    print('#')

                    # Debug print weights and biases.
                    # for param in self.net.parameters():
                    #     print(param.data)
                    #     print(param.shape)
                    # print('')

                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.3f' % (epoch + 1, i, avg_loss)
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                if i % 1000 == 0:
                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, data_len * epoch + i)

            self.net.eval()
            data = {}
            test_loss, problem_losses, _, _, _ = self.eval(epoch, do_print=False, debug=epoch % 10 == 0)
            problem_losses = {
                prob: problem_losses[i]
                for i, prob in enumerate(self.cfg['problems']['problems'])
            }
            data['loss_eval'] = test_loss
            data['problem_losses'] = problem_losses
            self.logger.log_eval_reverse(data, epoch)
            self.net.train()
        print('')

    def eval(self, epoch, do_print=True, debug=False):
        save_fpfn = self.cfg['problems']['problems'] == ["M2SAT"] or \
            self.cfg['problems']['problems'] == ["M3SAT"]

        n_classes = len(self.cfg['problems']['problems'])
        problem_losses = np.zeros(shape=(n_classes,), dtype=np.float32)
        n_class_labels = np.zeros(shape=(n_classes,), dtype=np.float32)

        wrong_regressions3 = 0
        tot3_fp = 0
        tot3_fn = 0
        tot3 = 0

        sse = 0
        ssm_mean = None
        n = 0

        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.lmdb_loader.test_data_loader):
            inputs, labels, prob = data
            outputs = self.net(inputs)
            loss = self.criterion(outputs, prob)

            sse += ((prob.numpy() - outputs[0].detach().numpy()) ** 2).sum()
            if ssm_mean is None:
                ssm_mean = prob.numpy()
            else:
                ssm_mean += prob.numpy()
            n += 1

            def print_debug(cutoff):
                output = outputs[0].detach().numpy()[:]

                if not save_fpfn:
                    # Useful for all except M2SAT.
                    output[np.where(output >= cutoff)] = 1.

                output = output.round() + 0

                if save_fpfn:
                    output[-1] = self.cfg['problems']['qubo_size']

                diff = output - prob[0].numpy()

                if not save_fpfn:
                    FP = np.count_nonzero(diff == 1.)
                    FN = np.count_nonzero(diff == -1.)
                    TOT = np.count_nonzero(prob[0].numpy())
                else:
                    FP = np.count_nonzero(diff >= 1)
                    FN = np.count_nonzero(diff <= -1)
                    TOT = np.count_nonzero(prob[0].numpy())

                # Debugging - What data/labels are problematic for the optimizer?
                # if diff.sum() < 0:
                #     a = outputs[0].detach().numpy()[:-1].reshape((4, 8, 8))
                #     b = prob[0].numpy()[:-1].reshape((4, 8, 8))

                #     import pdb; pdb.set_trace()

                return FP, FN, TOT

            if debug:
                FP, FN, TOT = print_debug(.3)
                if (FP + FN) > 0:
                    wrong_regressions3 += 1
                tot3_fp += FP
                tot3_fn += FN
                tot3 += TOT

            problem_losses[labels.item()] += loss
            n_class_labels[labels.item()] += 1

            total_loss += loss.item()
            if do_print and i % 1000 == 0:
                msg = '[%d] loss: %.3f' % (i, total_loss / (i + 1))
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()

        ssm_mean /= n
        ssm = 0
        for i, data in enumerate(self.lmdb_loader.test_data_loader):
            inputs, labels, prob = data
            ssm += ((prob.numpy() - ssm_mean) ** 2).sum()
        R2 = 1 - (sse / ssm)
        print("R2", R2)
        self.logger.log_custom_reverse_kpi("R2", R2, epoch)

        data_len = len(self.lmdb_loader.test_data_loader)

        for c in range(n_classes):
            problem_losses[c] /= n_class_labels[c]

        if debug:
            print('\n~~~~~ EVAL ~~~~~')
            print('.3', tot3_fp, tot3_fn, tot3, (tot3_fp + tot3_fn) / tot3, wrong_regressions3, data_len)
            self.logger.log_custom_reverse_kpi("FPFN_TOT_Ratio", (tot3_fp + tot3_fn) / tot3, epoch)

        return total_loss / data_len, problem_losses, tot3_fp, tot3_fn, tot3

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname, output_size):
        self.net = ReverseFCNet(self.cfg, output_size)
        self.net.load_state_dict(torch.load(model_fname))
