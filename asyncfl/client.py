import copy

import torch

from .dataloader import afl_dataset
from .network import MNIST_CNN, model_gradients


class Client:
    def __init__(self, pid, dataset_name: str) -> None:

        self.pid = pid
        self.dataset_name = dataset_name
        self.train_set, self.test_set = afl_dataset(dataset_name)
        self.device = torch.device('cpu')
        self.network = MNIST_CNN().to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.5)

    def get_pid(self):
        # print(f'My PID is: {self.pid}')
        return self.pid

    def print_pid_and_var(self, extra_text):
        print(f'My pid is {self.pid} + "{extra_text}"')

    def set_weights(self, weights):
        # print('Setting weigths')
        self.network.load_state_dict(copy.deepcopy(weights))

    def get_weights(self):
        return self.network.state_dict().copy()

    def get_gradients(self):
        return model_gradients(self.network)


    def train(self):
        try:
            inputs, labels = next(self.train_set)
        except StopIteration as _si:
            # Reload data
            self.train_set, self.test_set = afl_dataset(self.dataset_name)
            inputs, labels = next(self.train_set)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # print('Finished training')


    # def train(self):
    #     for i, (inputs, labels) in enumerate(self.train_set, 0):
    #         inputs, labels = inputs.to(self.device), labels.to(self.device)
    #         # zero the parameter gradients
    #         self.optimizer.zero_grad()
    #         outputs = self.network(inputs)
    #         loss = self.loss_function(outputs, labels)
    #         loss.backward()
    #         self.optimizer.step()
    #         print(outputs)
    #         break



