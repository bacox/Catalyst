import torch

from asyncfl import Client
from asyncfl.network import flatten_b, model_gradients


class RDCLient(Client):
    def __init__(self, pid, num_clients, dataset, model_name: str, sampler, sampler_args={}, learning_rate = 0.005, a_atk=0.2):
        super().__init__(pid, num_clients, dataset, model_name, sampler, sampler_args, learning_rate)
        self.a_atk = a_atk
        self.is_byzantine = True


    def get_gradients(self):
        # @TODO: Fix this, make it compatible with the super call!
        return [torch.add(torch.from_numpy(g), torch.randn_like(torch.from_numpy(g)).mul_(
            self.a_atk * torch.norm(torch.from_numpy(g), 2))).numpy() for g in model_gradients(self.network)]
    
    def get_gradient_vectors(self):
        # return [self.g_flat.cpu().numpy(), flatten_b(self.network).cpu().numpy(), self.local_age]
        g = self.g_flat.cpu()
        return [torch.add(g, torch.randn_like(g).mul_(self.a_atk * torch.norm(g, 2))).numpy(), flatten_b(self.network), self.lipschitz, self.convergance, self.local_age, self.is_byzantine]

class NGClient(Client):
    def __init__(self, pid, num_clients, dataset, model_name: str, sampler, sampler_args={}, learning_rate = 0.005, magnitude=10):
    # def __init__(self, pid, num_clients, dataset_name: str, model_name: str, magnitude=10):
        super().__init__(pid, num_clients, dataset, model_name, sampler, sampler_args, learning_rate)
        self.magnitude = magnitude


    def get_gradients(self):
        # gradients =  model_gradients(self.network)
        gradients, age = super().get_gradients()
        return [gradients * -1.0 * self.magnitude, age]
        # return self.g_flat.data.cpu().numpy()
        # return [x * -1 * self.magnitude for x in model_gradients(self.network)]

    def get_gradient_vectors(self):
        # return [self.g_flat.cpu().numpy(), flatten_b(self.network).cpu().numpy(), self.local_age]
        return [self.g_flat.cpu().numpy() * -1.0 * self.magnitude, flatten_b(self.network), self.lipschitz , self.local_age]
