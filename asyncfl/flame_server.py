import logging
from asyncfl.flame import flame
from asyncfl.server import Server, get_update, no_defense_update, parameters_dict_to_vector_flt
import math
import numpy as np


class FlameServer(Server):
    """
    Implementation of Flame Defense
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness
        self.model_history_dict = {}
        self.model_full_history = []

    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool): 
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
        update_params = get_update(weights, self.model_history[server_model_age])
        client_weight_vec = parameters_dict_to_vector_flt(weights)


        # Save model weights to dict
        self.model_history_dict[client_id] = (weights, update_params)
        self.model_full_history.append([weights, update_params])
        self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])

        if len(self.model_full_history) >= self.f * 2 + 1:
            # local_models = [item[0] for key, item in self.model_history_dict.items() if key != client_id] + [weights]
            # update_params_list = [item[1] for key, item in self.model_history_dict.items() if key != client_id] + [update_params]
            local_models = [item[0] for item in self.model_full_history]
            update_params_list = [item[1] for item in self.model_full_history]
            args_dict = {'num_users': 10, 'frac': 1.0, 'malicious': 0.3, 'wrong_mal': 0, 'right_ben': 0, 'turn':0}
            alpha = self.learning_rate / float(max(gradient_age, 1))
            global_model, accepted = flame(local_models, update_params_list, self.get_model_weights(), args_dict, alpha)
            self.set_weights(global_model)
            logging.info(f'[{is_byzantine},{accepted == is_byzantine:5}]Server accepts update? {accepted}, is byzantine ? {is_byzantine}')


            # Aggregate
            # alpha = self.learning_rate / float(max(gradient_age, 1))
            # self.set_weights(no_defense_update([update_params], self.get_model_weights(), alpha))
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