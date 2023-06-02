from asyncfl.server import Server, get_update, no_defense_update, parameters_dict_to_vector_flt
import math
import numpy as np


class SaSGD(Server):
    """
    Staleness Aware SGD: Implmentation of the n-softsync sgd algorithms where lambda == num_workers
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness

    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool): 
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
        update_params = get_update(weights, self.model_history[server_model_age])
        client_weight_vec = parameters_dict_to_vector_flt(weights)
        self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])

        # Aggregate
        alpha = self.learning_rate / float(max(gradient_age, 1))
        self.set_weights(no_defense_update([update_params], self.get_model_weights(), alpha))
        self.model_history.append(self.get_model_weights())
        self.incr_age()
        return self.get_model_weights()

    def client_update(self, _client_id: int, gradients: np.ndarray, client_lipschitz, client_convergence, gradient_age: int, is_byzantine: bool):
        # alpha = eta / tau_i_j
        alpha = self.learning_rate / float(gradient_age)
        for g in self.optimizer.param_groups:
            g['lr'] = alpha
        return super().client_update(_client_id, gradients, client_lipschitz, client_convergence, gradient_age, is_byzantine)

    
    # def client_update(self, _client_id: int, gradients: np.ndarray, gradient_age: int):
    #     model_staleness = (self.get_age() + 1) - gradient_age
    #     # print(f'Model_staleness={model_staleness}, damp_factor={dampening_factor(model_staleness, alpha=self.damp_alpha)}')
    #     gradients = gradients * dampening_factor(model_staleness, self.damp_alpha)
    #     return super().client_update(_client_id, gradients, gradient_age)