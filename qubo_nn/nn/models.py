import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


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


class Optimizer:
    def __init__(self, cfg, data, labels):
        self.cfg = cfg

        # Load cfg variables.
        lr = cfg['model']['lr']
        sgd_momentum = cfg['model']['optimizer_sgd_momentum']
        self.batch_size = cfg['model']['batch_size']
        self.n_epochs = cfg['model']['n_epochs']
        self.train_eval_split = cfg['model']['train_eval_split']

        # Set it all up.
        self.net = FCNet(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=lr, momentum=sgd_momentum
        )

        self._prep_data(data, labels)

    def _prep_data(self, data, labels):
        # TODO Move this to pipeline!! This does not belong here.
        tensor_x = torch.Tensor(data)
        tensor_y = torch.Tensor(labels)
        dataset = TensorDataset(tensor_x, tensor_y.long())

        train_size = int(self.train_eval_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        self.train_data_loader = DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        self.test_data_loader = DataLoader(
            test_dataset, batch_size=1
        )

    def train(self):
        self.net.train()
        for epoch in range(self.n_epochs):
            batch_loss = 0.
            for i, data in enumerate(self.train_data_loader, 0):
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
                if i % 10 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i, batch_loss / (i + 1)))

    def eval(self):
        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.test_data_loader, 0):
            inputs, labels = data
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            if i % 10 == 0:
                print('[%d] loss: %.3f' % (i, total_loss / (i + 1)))

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname):
        self.net = FCNet(self.cfg)
        self.net.load_state_dict(torch.load(model_fname))
