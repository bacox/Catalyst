import copy

import torch

from .dataloader import afl_dataset
from .network import MNIST_CNN, get_model_by_name, model_gradients, flatten, flatten_g


class Client:
    def __init__(self, pid, dataset_name: str, model_name: str) -> None:

        self.pid = pid
        self.dataset_name = dataset_name
        self.train_set, self.test_set = afl_dataset(dataset_name, use_iter=False, client_id=pid)
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.network = MNIST_CNN().to(self.device)
        self.network = get_model_by_name(model_name).to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.5)
        self.w_flat = flatten(self.network)
        self.g_flat = torch.zeros_like(self.w_flat)

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

    # def train(self, num_batches = -1):
    #     self.w_flat = flatten(self.network)
    #     self.g_flat = torch.zeros_like(self.w_flat)
    #     try:
    #         inputs, labels = next(iter(self.train_set))
    #     except StopIteration as _si:
    #         # Reload data
    #         self.train_set, self.test_set = afl_dataset(self.dataset_name)
    #         inputs, labels = next(iter(self.train_set))
    #     # print(len(self.train_set))
    #     inputs, labels = inputs.to(self.device), labels.to(self.device)
    #     # print(labels)
    #     # zero the parameter gradients
    #     self.optimizer.zero_grad()
    #     outputs = self.network(inputs)
    #     loss = self.loss_function(outputs, labels)
    #     loss.backward()
    #     flatten_g(self.network, self.g_flat)
    #     # self.g_flat.add_(self.w_flat)
    #     self.optimizer.step()

        # print('Finished training')


    def train(self, num_batches = -1):
        self.w_flat = flatten(self.network)
        g_flat_local = torch.zeros_like(self.w_flat)
        self.g_flat = torch.zeros_like(self.w_flat)
        self.optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(self.train_set):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            flatten_g(self.network, g_flat_local)
            self.g_flat += g_flat_local
            # print(self.g_flat)
            self.optimizer.step()
            if batch_idx == num_batches:
                break
            # print(self.g_flat)



    def get_gradients(self):
        # return model_gradients(self.network)
        return self.g_flat.data.cpu().numpy()
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



