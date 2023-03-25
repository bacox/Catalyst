from asyncfl.server import Server
import math
import numpy as np


class SaSGD(Server):
    """
    Staleness Aware SGD: Implmentation of the n-softsync sgd algorithms where lambda == num_workers
    """
    def __init__(self, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True) -> None:
        super().__init__(dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness


    def client_update(self, _client_id: int, gradients: np.ndarray, gradient_age: int):
        # alpha = eta / tau_i_j
        alpha = self.learning_rate / float(gradient_age)
        for g in self.optimizer.param_groups:
            g['lr'] = alpha
        return super().client_update(_client_id, gradients, gradient_age)

    
    # def client_update(self, _client_id: int, gradients: np.ndarray, gradient_age: int):
    #     model_staleness = (self.get_age() + 1) - gradient_age
    #     # print(f'Model_staleness={model_staleness}, damp_factor={dampening_factor(model_staleness, alpha=self.damp_alpha)}')
    #     gradients = gradients * dampening_factor(model_staleness, self.damp_alpha)
    #     return super().client_update(_client_id, gradients, gradient_age)