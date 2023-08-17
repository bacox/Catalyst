from asyncfl import Client
from asyncfl.network import flatten, flatten_b, model_gradients, unflatten
import torch
import logging

class NGClient(Client):
    def __init__(self, pid, num_clients, dataset, model_name: str, sampler, sampler_args={}, learning_rate = 0.005, magnitude=10):
    # def __init__(self, pid, num_clients, dataset_name: str, model_name: str, magnitude=10):
        super().__init__(pid, num_clients, dataset, model_name, sampler, sampler_args, learning_rate)
        self.magnitude = magnitude
        self.is_byzantine = True

    def train(self, num_batches=-1):
        logging.info(f'[Client {self.pid}] Running NG_Client training loop, magnitude={self.magnitude}')
        super().train(num_batches)
        weight_vector = flatten(self.network)
        inversed_weights = weight_vector * (self.g_flat * -1.0 * self.magnitude)
        unflatten(self.network, inversed_weights)


    def get_gradients(self):
        # gradients =  model_gradients(self.network)
        gradients, age = super().get_gradients()
        return [gradients * -1.0 * self.magnitude, age]
        # return self.g_flat.data.cpu().numpy()
        # return [x * -1 * self.magnitude for x in model_gradients(self.network)]

    def get_gradient_vectors(self):
        # return [self.g_flat.cpu().numpy(), flatten_b(self.network).cpu().numpy(), self.local_age]
        return [self.g_flat.cpu().numpy() * -1.0 * self.magnitude, flatten_b(self.network), self.lipschitz, self.convergance, self.local_age, self.is_byzantine]
