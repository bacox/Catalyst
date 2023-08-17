from asyncfl.server import Server, get_update, no_defense_update, parameters_dict_to_vector_flt
import math
import numpy as np
import logging

def fed_async_avg(parameters, parameters_server, alpha = 0.5):
    # if not sizes:
        # sizes = [1] * len(parameters)
    new_params = {}
    sum_size = 0
    # for client in parameters:
    for name in parameters.keys():
        # try:
        #     new_params[name].data = (parameters[name].data * alpha) + (parameters_server[name].data * (1-alpha))
        # except:
        new_params[name] = (parameters[name].detach().clone() * alpha) + (parameters_server[name].detach().clone() * (1-alpha))
        # new_params[name] = (parameters[name] * alpha) + (parameters_server[name] * (1-alpha))

        # sum_size += sizes[client]

    # for name in new_params:
    #     # @TODO: Is .long() really required?
    #     new_params[name].data = new_params[name].data.long() / sum_size
    return new_params

class FedAsync(Server):
    """
    Staleness Aware SGD: Implmentation of the n-softsync sgd algorithms where lambda == num_workers
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness
        self.alpha = 1

    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool): 
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
        staleness = 1 / float(self.age - gradient_age + 1)
        
        # s = 0.6
        # alpha = self.learning_rate / (s* float(max(staleness, 1)))
        alpha_t = self.alpha * staleness
        # for k in weights.keys():
        #     weights[k] = self.get_model_weights()[k] * (1-alpha_t) + weights[k] * alpha_t
        # self.set_weights(weights)
        self.set_weights(fed_async_avg(weights, self.get_model_weights(), alpha_t))


        # approx_gradient = get_update(weights, self.model_history[gradient_age])



        # # logging.info(f'Agg: s_age:{self.age}, c_age: {gradient_age}, u _age: {self.age - gradient_age}')
        # gradient_age = self.age - gradient_age
        # # update_params = get_update(weights, self.model_history[server_model_age])
        # # client_weight_vec = parameters_dict_to_vector_flt(weights)
        # # # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])
        # # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine])

        # # Aggregate
        # alpha = self.learning_rate / float(max(gradient_age, 1))
        # # alpha = self.learning_rate

        # self.set_weights(no_defense_update([weights], self.get_model_weights(), alpha))
        # self.set_weights(weights)
        logging.info(f'Agg with alpha: {alpha_t}')

        # Add approximate gradient to the current weights
        # logging.info(f'Approx gradient: {approx_gradient}')
        # self.set_weights(no_defense_update([approx_gradient], self.get_model_weights(), alpha))
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