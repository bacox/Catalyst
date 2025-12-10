from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt
import math
import numpy as np
import logging

class SaSGD(Server):
    """
    Staleness Aware SGD: Implmentation of the n-softsync sgd algorithms where lambda == num_workers
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness


    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> np.ndarray:
        logging.info(f'SaSGD dict_vector update of client {client_id}')
        gradient_age = self.age - gradient_age
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0

        approx_grad = weight_vec - self.model_history[server_model_age] # Gradient approximation

        alpha = self.learning_rate / float(max(gradient_age, 1))
        logging.info(f'SaSGD agregation with alpha: {alpha}')

        updated_model_vec = no_defense_vec_update([approx_grad], self.get_model_dict_vector(), server_rl=alpha)
        self.model_history.append(updated_model_vec)
        self.load_model_dict_vector(updated_model_vec)
        self.incr_age()
        return updated_model_vec.copy()


        self.set_weights(no_defense_update([approx_gradient], self.get_model_weights(), alpha))
        self.model_history.append(self.get_model_weights())
        self.incr_age()
        return self.get_model_weights()


        return super().client_weight_dict_vec_update(client_id, weight_vec, gradient_age, is_byzantine)

    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool): 
        """
        SA-sgd update the weights with staleness?

        SGD messes with the batch normalization in DNNs

        One solution might be to split the batch normalization from the sgd part

        A second option might be 
        """
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0

        approx_gradient = get_update(weights, self.model_history[gradient_age])



        # logging.info(f'Agg: s_age:{self.age}, c_age: {gradient_age}, u _age: {self.age - gradient_age}')
        gradient_age = self.age - gradient_age
        # update_params = get_update(weights, self.model_history[server_model_age])
        # client_weight_vec = parameters_dict_to_vector_flt(weights)
        # # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])
        # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine])

        # Aggregate
        alpha = self.learning_rate / float(max(gradient_age, 1))
        # alpha = self.learning_rate

        # self.set_weights(no_defense_update([weights], self.get_model_weights(), alpha))
        # self.set_weights(weights)
        logging.info(f'Agg with alpha: {alpha}')

        # Add approximate gradient to the current weights
        # logging.info(f'Approx gradient: {approx_gradient}')
        self.set_weights(no_defense_update([approx_gradient], self.get_model_weights(), alpha))
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