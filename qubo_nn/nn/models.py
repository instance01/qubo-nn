import sys
import pickle
import itertools
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

import qubovert
import numpy as np
from dwave_qbsolv import QBSolv


torch.set_printoptions(sci_mode=False)


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


class A3AutoEncoderFCNet(nn.Module):
    def __init__(self, cfg):
        super(A3AutoEncoderFCNet, self).__init__()
        self.input_size = cfg['problems']['qubo_size'] ** 2
        n_layers = cfg['model']['fc_sizes'][0]
        n_layers2 = 1
        if len(cfg['model']['fc_sizes']) > 1:
            n_layers2 = cfg['model']['fc_sizes'][1]

        encoder_net = []
        for _ in range(n_layers):
            encoder_net.append(nn.Linear(self.input_size, self.input_size))
            encoder_net.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*encoder_net[:-1])

        decoder_net = []
        for _ in range(n_layers2):
            decoder_net.append(nn.Linear(self.input_size, self.input_size))
            decoder_net.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*decoder_net[:-1])

        net = [
            self.encoder,
            self.decoder,
        ]
        self.fc_net = nn.Sequential(*net)

    def predict_encode(self, x):
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        # return nn.LeakyReLU()(x)  # TODO Hm..
        return x

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        size = int(self.input_size ** .5)
        return x.reshape(-1, size, size)


class R1AutoEncoderFCNet(nn.Module):
    def __init__(self, cfg):
        super(R1AutoEncoderFCNet, self).__init__()
        self.input_size = cfg['problems']['qubo_size'] ** 2

        self.encoder = nn.Linear(self.input_size, self.input_size)
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_size, self.input_size),
        #     nn.Linear(self.input_size, self.input_size)
        # )
        self.decoder = nn.Linear(self.input_size, self.input_size)
        net = [
            self.encoder,
            # nn.LeakyReLU(),
            self.decoder,
            # nn.LeakyReLU()
        ]
        self.fc_net = nn.Sequential(*net)

    def predict_encode(self, x):
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        # return nn.LeakyReLU()(x)  # TODO Hm..
        return x

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        size = int(self.input_size ** .5)
        return x.reshape(-1, size, size)


class R2AutoEncoderFCNet(nn.Module):
    def __init__(self, cfg):
        super(R2AutoEncoderFCNet, self).__init__()
        self.input_size = cfg['problems']['qubo_size'] ** 2

        # self.encoder = nn.Linear(self.input_size, self.input_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.Linear(self.input_size, self.input_size)
        )
        self.decoder = nn.Linear(self.input_size, self.input_size)
        net = [
            self.encoder,
            # nn.LeakyReLU(),
            self.decoder,
            nn.LeakyReLU()
        ]
        self.fc_net = nn.Sequential(*net)

        sol_net = [
            nn.Linear(self.input_size, self.input_size),
            nn.LeakyReLU()
        ]
        self.sol_fc = nn.Sequential(*sol_net)
        self.sol_head = nn.Linear(self.input_size, int(self.input_size ** .5))

    def predict_encode(self, x):
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        # return nn.LeakyReLU()(x)  # TODO Hm..
        return x

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        sol = self.sol_head(self.sol_fc(x))

        size = int(self.input_size ** .5)
        return x.reshape(-1, size, size), sol


class ReverseFCNet(nn.Module):
    def __init__(self, cfg, output_size):
        super(ReverseFCNet, self).__init__()
        input_size = cfg['problems']['qubo_size'] ** 2

        self.use_cnn = cfg['model'].get('use_cnn', False)
        self.cnn_cfg = cfg['model'].get('cnn_cfg', [])
        print(cfg['model'])

        self.is_qa = cfg['problems']['problems'] == ['QA']
        self.is_qk = cfg['problems']['problems'] == ['QK']
        self.use_special_norm = False
        if self.is_qk:
            self.use_special_norm = cfg['problems']['QK'].get('special_norm', False)
            if self.use_special_norm:
                size = cfg['problems']['QK']['size']
                input_size = 2 * size ** 2 + size

        # QA has a simplified input space.
        if self.is_qa:
            n = int((input_size ** .5) ** .5)
            input_size = int((n * (n - 1) / 2) ** 2)

        activation_type = cfg['model']['activation']
        if activation_type == "ReLU":
            activation_cls = nn.ReLU
        elif activation_type == "ELU":
            activation_cls = nn.ELU
        elif activation_type == "LeakyReLU":
            activation_cls = nn.LeakyReLU

        fc_sizes = cfg['model']['fc_sizes'] + [output_size]

        if self.use_cnn:
            net = []
            # E.g.: [["cnn", 1, 20, 8], ["max_pool"] ["flatten"], ["fc", 64]]
            for item in self.cnn_cfg:
                if item[0] == "cnn":
                    net.append(nn.Conv2d(*item[1:]))
                    net.append(activation_cls())
                elif item[0] == "max_pool":
                    net.append(nn.MaxPool2d(*item[1:]))
                elif item[0] == "flatten":
                    net.append(nn.Flatten(1))
                elif item[0] == "fc":
                    net.append(nn.Linear(*item[1:]))
                    net.append(activation_cls())
            # TODO: Currently we assume we end with ReLU (or so).
            net.pop(-1)
            self.fc_net = nn.Sequential(*net)
        else:
            net = []
            last_fc_size = input_size
            for size in fc_sizes:
                net.append(nn.Linear(last_fc_size, size))
                net.append(activation_cls())
                last_fc_size = size

            net.pop(-1)
            self.fc_net = nn.Sequential(*net)
        print(self.fc_net)
        n_params = sum(
            p.numel() for p in self.fc_net.parameters() if p.requires_grad
        )
        print("Number of trainable params:", n_params)

    def forward(self, x):
        if self.use_cnn:
            x = torch.unsqueeze(x, 1)  # We have just one channel.
        if not self.is_qa and not self.use_special_norm and not self.use_cnn:
            x = torch.flatten(x, 1)
        return self.fc_net(x)


class QbsolvFCNet(nn.Module):
    def __init__(self, cfg, output_size):
        super(QbsolvFCNet, self).__init__()
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

        n_params = sum(
            p.numel() for p in self.fc_net.parameters() if p.requires_grad
        )
        print("Number of trainable params:", n_params)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc_net(x)


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

        use_qa_loss = self.cfg['problems']['QA'].get('use_special_loss', False)

        self.net.train()
        data_len = len(self.lmdb_loader.train_data_loader)
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.lmdb_loader.train_data_loader):
                inputs, labels, problem = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs.float())

                if use_qa_loss:
                    loss = 0
                    for input_, output, prob in zip(inputs, outputs, problem):
                        len_ = int((len(output) / 2) ** .5)
                        o_x = output.reshape((2, len_, len_))[0][np.triu_indices(len_, 1)]
                        o_y = output.reshape((2, len_, len_))[1][np.triu_indices(len_, 1)]
                        half = len(o_x)
                        for k, x in enumerate(o_x):
                            for m, y in enumerate(o_y):
                                loss += (x * y - input_[k * half + m]) ** 2
                else:
                    loss = self.criterion(outputs, problem.float())
                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 1000 == 0:
                    X = problem.float().tolist()[-1]
                    Y = outputs.detach().tolist()[-1]
                    # print('')
                    # for m in range(len(X)):
                    #     if X[m] > 0 or X[m] < 0:
                    #         print(X[m], round(Y[m], 3))
                    # # print([round(x, 2) for x in outputs.detach().tolist()[-1][:100]])
                    # # print(problem.float().tolist()[-1][:100])
                    # print('#')

                    # for m in range(len(X)):
                    #     if X[m] > 0 or X[m] < 0:
                    #         print(X[m], round(Y[m], 3))

                    print([round(x, 2) for x in outputs.detach().tolist()[-1][:256]])
                    print([round(x, 2) for x in problem.float().tolist()[-1][:256]])
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
            test_loss, _, _, _ = self.eval(epoch, do_print=False, debug=epoch % 10 == 0)
            data['loss_eval'] = test_loss
            self.logger.log_eval_reverse(data, epoch)
            self.net.train()
        print('')

    def eval(self, epoch, do_print=True, debug=False):
        save_fpfn = self.cfg['problems']['problems'] == ["M2SAT"] or \
            self.cfg['problems']['problems'] == ["M3SAT"]

        use_qa_loss = self.cfg['problems']['QA'].get('use_special_loss', False)
        is_qa = self.cfg['problems']['problems'] == ["QA"]

        n_classes = len(self.cfg['problems']['problems'])

        wrong_regressions3 = 0
        tot3_fp = 0
        tot3_fn = 0
        tot3 = 0

        sse = 0
        ssm_mean = None
        n = 0

        max_mistake_loss = 0.
        max_mistake = None

        debugdata = []

        res_n = 0
        # residuals_neg = 0.
        # residuals_pos = 0.
        residuals_neg_arr1 = None
        residuals_pos_arr1 = None
        residuals_neg_arr2 = None
        residuals_pos_arr2 = None

        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.lmdb_loader.test_data_loader):
            inputs, labels, prob = data
            outputs = self.net(inputs.float())

            if use_qa_loss:
                loss = 0
                for input_, output in zip(inputs, outputs):
                    len_ = int((len(output) / 2) ** .5)
                    o_x = output.reshape((2, len_, len_))[0][np.triu_indices(len_, 1)]
                    o_y = output.reshape((2, len_, len_))[1][np.triu_indices(len_, 1)]
                    half = len(o_x)
                    for k, x in enumerate(o_x):
                        for m, y in enumerate(o_y):
                            loss += (x * y - input_[k * half + m]) ** 2
                            if i == 100:  # 100th sample
                                print(x, y, x * y, input_[k * half + m])
                if i == 100:
                    print(loss)
            else:
                loss = self.criterion(outputs, prob)

                if is_qa:
                    half = len(outputs[0]) // 2
                    residuals = outputs[0].detach().numpy() - prob[0].detach().numpy()
                    first_half = residuals[:half]
                    sec_half = residuals[half:]
                    if first_half.mean() < -.01 and sec_half.mean() > .01:
                        # residuals_neg1 += first_half.mean()
                        # residuals_pos1 += sec_half.mean()
                        if residuals_neg_arr1 is None:
                            residuals_neg_arr1 = first_half
                        else:
                            residuals_neg_arr1 += first_half
                        if residuals_pos_arr1 is None:
                            residuals_pos_arr1 = sec_half
                        else:
                            residuals_pos_arr1 += sec_half
                        res_n += 1
                    if first_half.mean() > .01 and sec_half.mean() < -.01:
                        # residuals_pos += first_half.mean()
                        # residuals_neg += sec_half.mean()
                        if residuals_neg_arr2 is None:
                            residuals_neg_arr2 = sec_half
                        else:
                            residuals_neg_arr2 += sec_half
                        if residuals_pos_arr2 is None:
                            residuals_pos_arr2 = first_half
                        else:
                            residuals_pos_arr2 += first_half
                        res_n += 1

            if loss > max_mistake_loss:
                max_mistake = (inputs, labels, prob, outputs)
                max_mistake_loss = loss
            # if epoch >= 10:
            #     debugdata.append((loss.detach().item(), inputs.detach().numpy(), outputs.detach().numpy(), prob.numpy()))

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

                # TODO Used for comparison plot (QUBO, problem/label, predicted problem)
                # import matplotlib.pyplot as plt
                # xxx = inputs
                # yyy = prob
                # fig, axs = plt.subplots(1, 3, figsize=(12, 6))
                # # im = ax.imshow(xxx.numpy(), cmap=cmap_mod, vmin=0, vmax=1)
                # min_ = min([xxx.numpy().min(), yyy.detach().numpy()[0][:-1].min(), output[:-1].min()])
                # max_ = min([xxx.numpy().max(), yyy.detach().numpy()[0][:-1].max(), output[:-1].max()])
                # im = axs[0].imshow(xxx.numpy()[0], vmin=min_, vmax=max_)
                # im2 = axs[1].imshow(yyy.detach().numpy()[0][:-1].reshape((16,16)), vmin=min_, vmax=max_)
                # im3 = axs[2].imshow(output[:-1].reshape((16,16)), vmin=min_, vmax=max_)
                # axs[0].set_xlabel('QUBO', rotation=0, va="top", fontsize=9)
                # axs[1].set_xlabel('Problem (label)', rotation=0, va="top", fontsize=9)
                # axs[2].set_xlabel('Predicted Problem (output)', rotation=0, va="top", fontsize=9)
                # plt.show()
                # # import pdb; pdb.set_trace()

                return FP, FN, TOT

            # if debug:
            #     FP, FN, TOT = print_debug(.3)
            #     if (FP + FN) > 0:
            #         wrong_regressions3 += 1
            #     tot3_fp += FP
            #     tot3_fn += FN
            #     tot3 += TOT

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
        print(" ", sse, ssm)
        print("R2", R2)
        # print("MAX MISTAKE", max_mistake_loss, "\n", max_mistake)

        # if epoch >= 10:
        #     with open("qa_debug.pickle", "wb+") as f:
        #         pickle.dump(debugdata, f)
        #     import pdb; pdb.set_trace()
        self.logger.log_custom_reverse_kpi("R2", R2, epoch)

        data_len = len(self.lmdb_loader.test_data_loader)

        if is_qa:
            self.logger.log_custom_reverse_kpi("Res_Tot", res_n / data_len, epoch)
            if residuals_neg_arr1 is not None:
                print("res_neg1", (residuals_neg_arr1 / res_n).tolist())
            if residuals_pos_arr1 is not None:
                print("res_pos1", (residuals_pos_arr1 / res_n).tolist())
            if residuals_neg_arr2 is not None:
                print("res_neg2", (residuals_neg_arr2 / res_n).tolist())
            if residuals_pos_arr2 is not None:
                print("res_pos2", (residuals_pos_arr2 / res_n).tolist())
            print("res_tot", res_n / data_len)

        if debug:
            print('\n~~~~~ EVAL ~~~~~')
            # print('.3', tot3_fp, tot3_fn, tot3, (tot3_fp + tot3_fn) / tot3, wrong_regressions3, data_len)
            # self.logger.log_custom_reverse_kpi("FPFN_TOT_Ratio", (tot3_fp + tot3_fn) / tot3, epoch)

        return total_loss / data_len, tot3_fp, tot3_fn, tot3

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname, output_size):
        self.net = ReverseFCNet(self.cfg, output_size)
        self.net.load_state_dict(torch.load(model_fname))


class RNN(nn.Module):
    def __init__(self, cfg, output_size):
        super(RNN, self).__init__()
        self.cfg = cfg

        input_size = cfg['problems']['qubo_size']

        activation_type = cfg['model']['activation']
        non_linearity = 'tanh'
        if activation_type == "ReLU":
            non_linearity = 'relu'

        # self.rnn = nn.RNN(
        #     input_size=input_size,
        #     hidden_size=cfg['model']['fc_sizes'][0],
        #     num_layers=len(cfg['model']['fc_sizes']),
        #     nonlinearity=non_linearity,
        #     batch_first=True,
        # )
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg['model']['fc_sizes'][0],
            num_layers=len(cfg['model']['fc_sizes']),
            batch_first=True,
        )

        self.out = nn.Linear(cfg['model']['fc_sizes'][-1], input_size)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # 1 stands for n_layers
        hidden_dim = self.cfg['model']['fc_sizes'][0]
        hidden = (
            weight.new(1, batch_size, hidden_dim).zero_(),
            weight.new(1, batch_size, hidden_dim).zero_()
        )
        return hidden


class RNNOptimizer(Optimizer):
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
        self.net = RNN(cfg, output_size)
        self.criterion = nn.MSELoss()
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

            h_state = self.net.init_hidden(self.batch_size)

            for i, data in enumerate(self.lmdb_loader.train_data_loader):
                inputs, labels, problem = data

                problem = problem[:, :-1].reshape((-1, 16, 16))

                self.optimizer.zero_grad()

                h_state = tuple([e.data for e in h_state])
                outputs, h_state = self.net(inputs, h_state)

                loss = self.criterion(outputs, problem.float())
                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 100 == 0:
                    X = problem.float()[-1].flatten().tolist()[:200]
                    Y = outputs.detach()[-1].flatten().tolist()[:200]
                    print('')
                    for m in range(200):
                        if X[m] > 0 or X[m] < 0:
                            print(X[m], round(Y[m], 3))
                    print('#')

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
            outputs, h_state = self.net(inputs, None)

            prob = prob[:, :-1].reshape((-1, 16, 16))

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
            prob = prob[:, :-1].reshape((-1, 16, 16))
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


def solve_qubo2(batch):
    qubo_size = 8  # TODO Hardcoded!

    sols = []
    for item in batch:
        Q = qubovert.utils.matrix_to_qubo(item.reshape(qubo_size, qubo_size))
        sol = Q.solve_bruteforce(all_solutions=False)
        sol_ = [0 for _ in range(qubo_size)]
        for k, v in sol.items():
            sol_[k] = v
        sols.append(sol_)
    return torch.FloatTensor(sols)


class QBSolvFunction(torch.autograd.Function):
    """Reference: Differentiable blackbox combinatorial solvers."""
    @staticmethod
    def forward(ctx, weights, lambda_val):
        # weights is the 'suggested' QUBO.
        ctx.weights = weights.detach().cpu().numpy()
        ctx.lambda_val = lambda_val
        ctx.suggested_sol = solve_qubo2(ctx.weights).numpy()
        return torch.from_numpy(ctx.suggested_sol).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_sol.shape
        # TODO repeat (1,8) added!
        grad_output_numpy = grad_output.repeat((1, 8)).detach().cpu().numpy()
        weights_prime = ctx.weights + ctx.lambda_val * grad_output_numpy
        better_sol = solve_qubo2(weights_prime).numpy()
        gradient = -(ctx.suggested_sol - better_sol) / ctx.lambda_val
        return torch.from_numpy(gradient).to(grad_output.device).repeat((1, 8)), None


class A3Optimizer:
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
        self.n_problems = len(cfg['problems']['problems'])

        # Set it all up.
        self.nets = []
        for _ in range(self.n_problems):
            self.nets.append(A3AutoEncoderFCNet(cfg))
        self.criterion = nn.MSELoss()
        self.optimizers = []
        for i in range(self.n_problems):
            self.optimizers.append(optim.SGD(
                self.nets[i].parameters(), lr=lr, momentum=sgd_momentum
            ))

        # TODO Hardcoded!!
        self.batch_size = 10

        self.qubo_size = self.cfg['problems']['qubo_size'] 
        print(self.nets)

    def solve_qubo(self, batch):
        # NOTE: Unused function.
        # TODO Don't solve QUBOs on the fly.. This needs fixing.
        ret_batch = []
        qb = QBSolv()
        for item in batch:
            item = item.reshape(self.qubo_size, self.qubo_size)

            Q = {}
            # We can assume a quadratic matrix.
            for i in range(item.shape[0]):
                for j in range(item.shape[1]):
                    if item[i][j] != 0:
                        Q[(i, j)] = item[i][j]
            response = qb.sample_qubo(Q, num_repeats=100)
            ret = [0] * self.qubo_size
            for k, v in response.samples()[0].items():
                ret[k] = v
            ret_batch.append(ret)
        return torch.FloatTensor(ret_batch)

    def train(self):
        use_qbsolv_loss = self.cfg['model'].get('use_qbsolv_loss', False)
        use_similarity_loss = self.cfg['model'].get('use_similarity_loss', False)

        [net.train() for net in self.nets]
        for epoch in range(self.n_epochs):
            batch_loss = 0.

            all_inputs = list(self.lmdb_loader.train_data_loader)

            len_ = len(all_inputs[0][0][0])
            for i in range(len_):
                chosen_data = []
                for problem_specific_input, labels in all_inputs:
                    curr_data = []
                    curr_labels = []
                    for j in range(self.batch_size):
                        idx = np.random.randint(0, len_)
                        curr_data.append(problem_specific_input[0][idx].unsqueeze(0))
                        curr_labels.append(labels[0][idx])

                    tensor = torch.Tensor(
                        self.batch_size,
                        self.qubo_size,
                        self.qubo_size
                    )
                    torch.cat(curr_data, out=tensor)

                    tensor2 = torch.Tensor(self.batch_size, self.qubo_size)
                    torch.cat(curr_labels, out=tensor2)
                    tensor2 = tensor2.reshape((10, 8))

                    chosen_data.append((tensor, tensor2))

                loss = 0
                latent_outputs = []

                debug_losses = []

                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                for j, (inputs, labels) in enumerate(chosen_data):
                    outputs = self.nets[j](inputs)
                    latent_output = self.nets[j].predict_encode(inputs)
                    latent_outputs.append(latent_output)
                    qubo_loss = self.criterion(outputs, inputs)
                    debug_losses.append(qubo_loss.item())

                    if use_qbsolv_loss:
                        # true_sol = solve_qubo2(inputs)  # TODO: This should happen in gendata phase.
                        true_sol = labels
                        suggested_sol = QBSolvFunction.apply(latent_output, 10.0)
                        # print(suggested_sol)
                        # print(true_sol)
                        qbsolv_loss = self.criterion(suggested_sol, true_sol)
                        loss += qbsolv_loss
                        debug_losses.append(qbsolv_loss.item())

                    loss += qubo_loss
                    debug_losses.append(qubo_loss.item())

                if use_similarity_loss:
                    for l1, l2 in itertools.combinations(latent_outputs, 2):
                        # similarity_loss = self.criterion(l1, l2) / len_
                        similarity_loss = self.criterion(l1, l2)
                        loss += similarity_loss
                        debug_losses.append(similarity_loss.item())

                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()

                batch_loss += loss.item() * self.batch_size
                if i % 100 == 0:
                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.3f' % (epoch + 1, i, avg_loss)
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()
                    print(debug_losses)
                if i % 100 == 0:
                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, len_ * epoch + i)
        print('')

    def eval(self, do_print=True):
        # TODO
        return {}, 0, {}, {}

    def save(self, model_fname):
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), 'models/' + model_fname + "-" + str(i))

    def load(self, model_fname):
        self.net = A3AutoEncoderFCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))


class Resistance1:
    """Reference: Differentiable blackbox combinatorial solvers."""
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
        self.n_problems = len(cfg['problems']['problems'])

        # Set it all up.
        self.net = R1AutoEncoderFCNet(cfg)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=sgd_momentum)

        # TODO Hardcoded!
        self.batch_size = 2

        self.qubo_size = self.cfg['problems']['qubo_size'] 

        print(self.net)
        print(self.batch_size)

    def train(self):
        lambda_ = self.cfg["model"]["lambda"]

        self.net.train()
        for epoch in range(self.n_epochs):
            batch_loss = 0.

            len_ = len(self.lmdb_loader.train_data_loader)

            for i, (inputs, _) in enumerate(self.lmdb_loader.train_data_loader):
                loss = 0
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                latent_output = self.net.predict_encode(inputs)
                qubo_loss = self.criterion(outputs, inputs)

                true_sol = solve_qubo2(inputs)  # TODO: This should happen in gendata phase.
                suggested_sol = QBSolvFunction.apply(latent_output, lambda_)
                qbsolv_loss = self.criterion(suggested_sol, true_sol)

                if i % 500 == 0:
                    print(inputs)
                    print(outputs)
                    print(latent_output)

                loss += qubo_loss
                loss += qbsolv_loss

                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item() * self.batch_size
                if i % 10 == 0:
                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.3f | %.4f %.4f' % (epoch + 1, i, avg_loss, qubo_loss.item(), qbsolv_loss.item())
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                if i % 100 == 0:
                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, len_ * epoch + i)
        print('')

    def eval(self, do_print=True):
        # TODO
        return {}, 0, {}, {}

    def save(self, model_fname):
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), 'models/' + model_fname + "-" + str(i))

    def load(self, model_fname):
        self.net = R1AutoEncoderFCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))


class Resistance2:
    """Two heads."""
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
        self.n_problems = len(cfg['problems']['problems'])

        # Set it all up.
        self.net = R2AutoEncoderFCNet(cfg)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=sgd_momentum)

        # TODO Hardcoded!
        self.batch_size = 2
        self.qubo_size = self.cfg['problems']['qubo_size'] 
        print(self.net)

    def train(self):
        self.net.train()
        for epoch in range(self.n_epochs):
            batch_loss = 0.

            len_ = len(self.lmdb_loader.train_data_loader)

            for i, (inputs, _) in enumerate(self.lmdb_loader.train_data_loader):
                loss = 0
                self.optimizer.zero_grad()

                outputs, pred_sol = self.net(inputs)
                latent_output = self.net.predict_encode(inputs)
                qubo_loss = self.criterion(outputs, inputs)

                true_sol = solve_qubo2(inputs)  # TODO: This should happen in gendata phase.
                # pred_sol = solve_qubo2(latent_output)
                qbsolv_loss = self.criterion(pred_sol, true_sol) / 10.

                if i % 500 == 0:
                    print(inputs)
                    print(outputs)
                    print(latent_output)

                loss += qubo_loss
                loss += qbsolv_loss

                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item() * self.batch_size
                if i % 10 == 0:
                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.4f | %.4f %.4f' % (epoch + 1, i, avg_loss, qubo_loss.item(), qbsolv_loss.item())
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                if i % 100 == 0:
                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, len_ * epoch + i)
        print('')

    def eval(self, do_print=True):
        # TODO
        return {}, 0, {}, {}

    def save(self, model_fname):
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), 'models/' + model_fname + "-" + str(i))

    def load(self, model_fname):
        self.net = R1AutoEncoderFCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))


class QbsolvOptimizer(Optimizer):
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
        self.qubo_size = cfg['problems']['qubo_size']

        # Set it all up.
        self.net = QbsolvFCNet(cfg, self.qubo_size)
        self.criterion = nn.MSELoss()
        if cfg['model']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=lr, momentum=sgd_momentum
            )
        elif cfg['model']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=lr
            )

    def train(self):
        self.net.train()
        data_len = len(self.lmdb_loader.train_data_loader)
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.lmdb_loader.train_data_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs.float())

                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 1000 == 0:
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
            test_loss, _, _, _ = self.eval(epoch, do_print=False, debug=epoch % 10 == 0)
            data['loss_eval'] = test_loss
            self.logger.log_eval_reverse(data, epoch)
            self.net.train()
        print('')

    def eval(self, epoch, do_print=True, debug=False):
        sse = 0
        ssm_mean = None
        n = 0

        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.lmdb_loader.test_data_loader):
            inputs, labels = data
            labels = labels.float()
            outputs = self.net(inputs.float())

            loss = self.criterion(outputs, labels)

            sse += ((labels.numpy() - outputs[0].detach().numpy()) ** 2).sum()
            if ssm_mean is None:
                ssm_mean = labels.numpy()
            else:
                ssm_mean += labels.numpy()
            n += 1

            total_loss += loss.item()
            if do_print and i % 1000 == 0:
                msg = '[%d] loss: %.3f' % (i, total_loss / (i + 1))
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()

        ssm_mean /= n
        ssm = 0
        for i, data in enumerate(self.lmdb_loader.test_data_loader):
            inputs, labels = data
            ssm += ((labels.numpy() - ssm_mean) ** 2).sum()
        R2 = 1 - (sse / ssm)
        print(" ", sse, ssm)
        print("R2", R2)

        self.logger.log_custom_reverse_kpi("R2", R2, epoch)

        data_len = len(self.lmdb_loader.test_data_loader)

        return total_loss / data_len, 0, 0, 0

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname):
        self.net = ReverseFCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))
