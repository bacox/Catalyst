from asyncfl.kardam import Kardam, dampening_factor
from asyncfl.network import flatten
from asyncfl.server import Server
import numpy as np
import torch

from asyncfl.util import compute_lipschitz_simple


class Telerig(Kardam):

    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, damp_alpha: float = 0.2) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate, damp_alpha)

    
    def client_update(self, _client_id: int, gradients: np.ndarray, client_lipschitz, gradient_age: int):
        self.lips[_client_id] = client_lipschitz.numpy()
        model_staleness = (self.get_age() + 1) - gradient_age
        grads = torch.from_numpy(gradients)
        self.grad_history[self.age] = grads.clone()
        grads_dampened = grads * dampening_factor(model_staleness, self.damp_alpha)
        prev_model = flatten(self.network).detach().clone().cpu()
        current_model = flatten(self.network).detach().clone().cpu() * self.learning_rate * grads
        self.k_pt = compute_lipschitz_simple(grads_dampened, self.prev_gradients, current_model, prev_model)
        self.prune_grad_history(size=4)
        if self.lipschitz_check(self.k_pt, self.hack):
            # call the apply gradients from the extended ps after gradient has been checked
            if self.frequency_check(_client_id):
                self.bft_telemetry["accepted"][_client_id]["values"].append([self.k_pt,self.get_age()])
                self.bft_telemetry["accepted"][_client_id]["total"] += 1
                # @TODO: Change this if it does not work!
                alpha = dampening_factor(model_staleness, self.damp_alpha)
                for g in self.optimizer.param_groups:
                    g['lr'] = alpha
                self.aggregate(grads)
                return flatten(self.network)
            else:
                self.bft_telemetry["rejected"][_client_id]["values"].append([self.k_pt,self.get_age()])
                self.bft_telemetry["rejected"][_client_id]["total"] += 1
                return flatten(self.network)
        else:
            self.bft_telemetry["rejected"][_client_id]["values"].append([self.k_pt,self.get_age()])
            self.bft_telemetry["rejected"][_client_id]["total"] += 1
            return flatten(self.network)
