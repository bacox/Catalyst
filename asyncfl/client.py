import copy
from typing import List

import torch
import numpy as np
import logging
from asyncfl.util import compute_convergance, compute_lipschitz_simple
from .dataloader import afl_dataloader, afl_dataset
from .network import MNIST_CNN, flatten_b, flatten_dict, get_model_by_name, model_gradients, flatten, flatten_g, unflatten, unflatten_dict
from torch.utils.data import DataLoader

def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))

class Client:
    def __init__(self, pid, num_clients, dataset, model_name: str, sampler, sampler_args={}, learning_rate = 0.005) -> None:

        self.pid = pid
        self.train_set = afl_dataloader(dataset, use_iter=False, client_id=pid, n_clients=num_clients, sampler=sampler, sampler_args=sampler_args)
        # self.train_set = afl_dataset(dataset_name, use_iter=True, client_id=pid, n_clients=num_clients, sampler=sampler, sampler_args=sampler_args)
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.network = MNIST_CNN().to(self.device)
        # self.network = get_model_by_name(model_name).to(self.device)
        # return
        self.network = get_model_by_name(model_name)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.5)
        self.w_flat = flatten(self.network)
        self.g_flat = torch.zeros_like(self.w_flat)
        self.local_age = 0
        # self.lipschitz = None
        self.lipschitz = 0
        self.convergance = 0
        self.is_byzantine = False

        self.prev_weights =  torch.zeros_like(self.w_flat)
        self.prev_gradients = torch.zeros_like(self.w_flat)
        self.prev_prev_gradients = torch.zeros_like(self.w_flat)

    def get_pid(self):
        return self.pid

    def print_pid_and_var(self, extra_text):
        print(f'My pid is {self.pid} + "{extra_text}"')

    def set_weights(self, weights, age: int):
        self.local_age = age
        self.network.load_state_dict(copy.deepcopy(weights))

    def get_weights(self):
        return self.network.state_dict().copy()
    

    def get_weight_vectors(self):
        # Make flat vector of paramters
        # Make flat vector of buffers
        return [flatten(self.network), flatten_b(self.network)]
    
    def get_gradient_vectors(self):
        # return [self.g_flat.cpu().numpy(), flatten_b(self.network).cpu().numpy(), self.local_age]
        return [self.g_flat.cpu().numpy(), flatten_b(self.network), self.lipschitz, self.convergance , self.local_age, self.is_byzantine]
    

    def set_weight_vectors(self, weights: np.ndarray, age):
        self.local_age = age
        unflatten(self.network, torch.from_numpy(weights).to(self.device))

    
    def get_model_dict_vector(self) -> np.ndarray:
        return flatten_dict(self.network).cpu().numpy()
    
    def get_model_dict_vector_t(self) ->torch.Tensor:
        return flatten_dict(self.network)
    
    def load_model_dict_vector(self, vec: np.ndarray):
        vec = torch.from_numpy(vec).to(self.device)
        unflatten_dict(self.network, vec)

    

    # GPU AUX FUNCTIONS
    def move_to_gpu(self):
        self.network.to(self.device)
        pass

    def move_to_cpu(self):
        self.network.to(torch.device('cpu'))

    # def train(self, num_batches = -1):
    #     self.optimizer.zero_grad()
    #     self.w_flat = flatten(self.network)
    #     self.g_flat = torch.zeros_like(self.w_flat)
    #     g_flat_local = torch.zeros_like(self.w_flat)
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
    #     outputs = self.network(inputs)
    #     loss = self.loss_function(outputs, labels)
    #     loss.backward()
    #     flatten_g(self.network, g_flat_local)
    #     g_flat_local = g_flat_local.detach().clone()
    #     # Not sure if we want to multiply against the learning rate already
    #     # g_flat_local.mul_(self.lr)
    #     self.g_flat.add_(g_flat_local)
    #     # self.g_flat.add_(self.w_flat.detach().clone())
    #     self.optimizer.step()
    #     # self.optimizer.zero_grad()

    #     # print('Finished training')


    def train(self, num_batches = -1, local_epochs = 1):
        # @TODO: Increment local_age
        logging.info(f'[Client {self.pid}] training')
        self.w_flat = flatten(self.network)

        prev_weights = self.w_flat.detach().clone()
        # prev_gradients = self.g_flat.detach().clone()

        self.prev_prev_gradients = self.prev_gradients.clone()
        self.prev_gradients = self.g_flat.detach().clone()


        g_flat_local = torch.zeros_like(self.w_flat)
        self.g_flat = torch.zeros_like(self.w_flat)
        self.optimizer.zero_grad()
        for ep in range(local_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.train_set):
                # print(f'[Client{self.pid}]:: {batch_idx}')
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                flatten_g(self.network, g_flat_local)
                # g_flat_local.g_flat.mul_(self.lr)
                self.g_flat.add_(g_flat_local)
                # print(self.g_flat)
                self.optimizer.step()
                # logging.info(f'[{self.pid}] loss: {loss}')
                if batch_idx == num_batches:
                    break

            # print(self.g_flat)
        current_weights = flatten(self.network)

        
        self.lipschitz = compute_lipschitz_simple(self.g_flat.cpu(), self.prev_gradients.cpu(), current_weights.cpu(), prev_weights.cpu())
        self.convergance = compute_convergance(self.g_flat, self.prev_gradients, self.prev_prev_gradients)



    def get_gradients(self):
        # return model_gradients(self.network)
        return [self.g_flat.cpu().detach().clone().numpy(), self.local_age]
        # return [self.g_flat.grad.data.cpu().detach().clone().numpy(), self.local_age]
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



