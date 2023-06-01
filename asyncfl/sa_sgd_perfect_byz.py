from asyncfl.server import Server
from asyncfl.server import flatten
import math
import numpy as np


class SaSGDPerfectByz(Server):
    """
    Staleness Aware SGD: Implmentation of the n-softsync sgd algorithms where lambda == num_workers.
    This server perfectly filters the byzantine nodes. This is by design to capture a "perfect" run of byzantine learning.
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness


    def client_update(self, _client_id: int, gradients: np.ndarray, client_lipschitz, client_convergence, gradient_age: int, is_byzantine: bool):
        # alpha = eta / tau_i_j
        alpha = self.learning_rate / float(gradient_age)
        for g in self.optimizer.param_groups:
            g['lr'] = alpha
        if is_byzantine:
            # Don't aggregate, just return current model
            return flatten(self.network)
        return super().client_update(_client_id, gradients, client_lipschitz, client_convergence, gradient_age, is_byzantine)

    
    # def client_update(self, _client_id: int, gradients: np.ndarray, gradient_age: int):
    #     model_staleness = (self.get_age() + 1) - gradient_age
    #     # print(f'Model_staleness={model_staleness}, damp_factor={dampening_factor(model_staleness, alpha=self.damp_alpha)}')
    #     gradients = gradients * dampening_factor(model_staleness, self.damp_alpha)
    #     return super().client_update(_client_id, gradients, gradient_age)