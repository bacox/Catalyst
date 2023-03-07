from asyncfl import Client
from asyncfl.network import model_gradients


class NGClient(Client):

    def __init__(self, pid, dataset_name: str, magnitude=10):
        super().__init__(pid, dataset_name)
        self.magnitude = magnitude


    def get_gradients(self):
        # gradients =  model_gradients(self.network)
        return [x * -1 * self.magnitude for x in model_gradients(self.network)]
