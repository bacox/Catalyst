import torch

from asyncfl import Client
from asyncfl.network import model_gradients


class RDCLient(Client):
    def __init__(self, pid, num_clients, dataset_name: str, model_name: str, a_atk=0.2):
        super().__init__(pid, num_clients, dataset_name, model_name)
        self.a_atk = a_atk


    def get_gradients(self):
        # @TODO: Fix this, make it compatible with the super call!
        return [torch.add(torch.from_numpy(g), torch.randn_like(torch.from_numpy(g)).mul_(
            self.a_atk * torch.norm(torch.from_numpy(g), 2))).numpy() for g in model_gradients(self.network)]