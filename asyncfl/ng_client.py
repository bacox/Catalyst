from asyncfl import Client
from asyncfl.network import flatten_b, model_gradients
import torch

class NGClient(Client):
    def __init__(self, pid, num_clients, dataset, model_name: str, sampler, sampler_args={}, learning_rate = 0.005, magnitude=10):
    # def __init__(self, pid, num_clients, dataset_name: str, model_name: str, magnitude=10):
        super().__init__(pid, num_clients, dataset, model_name, sampler, sampler_args, learning_rate)
        self.magnitude = magnitude
        self.is_byzantine = True


    def get_gradients(self):
        # gradients =  model_gradients(self.network)
        gradients, age = super().get_gradients()
        return [gradients * -1.0 * self.magnitude, age]
        # return self.g_flat.data.cpu().numpy()
        # return [x * -1 * self.magnitude for x in model_gradients(self.network)]

    def get_gradient_vectors(self):
        # return [self.g_flat.cpu().numpy(), flatten_b(self.network).cpu().numpy(), self.local_age]
        return [self.g_flat.cpu().numpy() * -1.0 * self.magnitude, flatten_b(self.network), self.lipschitz , self.local_age, self.is_byzantine]
