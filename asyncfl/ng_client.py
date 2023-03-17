from asyncfl import Client
from asyncfl.network import model_gradients
import torch

class NGClient(Client):

    def __init__(self, pid, num_clients, dataset_name: str, model_name: str, magnitude=10):
        super().__init__(pid, num_clients, dataset_name, model_name)
        self.magnitude = magnitude


    def get_gradients(self):
        # gradients =  model_gradients(self.network)
        return super().get_gradients() * -1.0 * self.magnitude
        # return self.g_flat.data.cpu().numpy()
        # return [x * -1 * self.magnitude for x in model_gradients(self.network)]
