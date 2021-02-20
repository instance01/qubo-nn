import sys
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, cfg):
        super(FCNet, self).__init__()
        input_size = cfg['problems']['qubo_size'] ** 2

        net = []
        last_fc_size = input_size
        for size in cfg['model']['fc_sizes']:
            net.append(nn.Linear(last_fc_size, size))
            net.append(nn.ReLU())
            last_fc_size = size

        self.fc_net = nn.Sequential(*net)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        return F.log_softmax(x, dim=1)


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
        print(data_len)
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.lmdb_loader.train_data_loader, 0):
                print(i)
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
                if i % 1000 == 0:
                    msg = '[%d, %5d] loss: %.3f' % (epoch + 1, i, batch_loss / (i + 1))
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                    data = {
                        "loss_train": batch_loss / (i + 1)
                    }
                    self.logger.log_train(data, data_len * epoch + i)

            self.net.eval()
            data = {}
            misclassifications, test_loss, tot_mc = self.eval(False)
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

        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.lmdb_loader.test_data_loader, 0):
            inputs, labels = data
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            if torch.argmax(outputs) != labels:
                misclassifications[labels.item()] += 1

            total_loss += loss.item()
            if do_print and i % 1000 == 0:
                msg = '[%d] loss: %.3f' % (i, total_loss / (i + 1))
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()

        data_len = len(self.lmdb_loader.test_data_loader)
        tot_mc = float(sum(misclassifications.values())) / data_len
        if do_print:
            print('\nMisclassification rate', tot_mc)
        return misclassifications, total_loss / data_len, tot_mc

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname):
        self.net = FCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))
