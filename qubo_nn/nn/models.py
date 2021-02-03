import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(20 ** 2, 10)
        self.fc2 = nn.Linear(10, 4)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNNet(nn.Module):
    pass  # TODO


class Optimizer:
    def __init__(self, data, labels):
        self.net = FCNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=0.001, momentum=0.9
        )
        tensor_x = torch.Tensor(data)
        tensor_y = torch.Tensor(labels)
        dataset = TensorDataset(tensor_x, tensor_y.long())
        self.data_loader = DataLoader(dataset, batch_size=100)
        self.n_epochs = 5

    def train(self):
        data_len = len(self.data_loader.dataset)
        for epoch in range(self.n_epochs):
            running_loss = 0.0

            for i, data in enumerate(self.data_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / data_len))
                    running_loss = 0.0

    def eval(self):
        pass  # TODO
