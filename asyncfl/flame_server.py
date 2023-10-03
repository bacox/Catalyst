import logging
from typing import List
from asyncfl.flame import flame, flame_v2, flame_v3
from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt
import math
import numpy as np


class FlameServer(Server):
    """
    Implementation of Flame Defense
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, mitigate_staleness = True, hist_size = 3,min_cluster_size=3) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.mitigate_staleness = mitigate_staleness
        self.model_history_dict = {}
        self.model_full_history = []
        self.hist_size = hist_size
        self.min_cluster_size = min_cluster_size

    
    def aggregate_sync(self, params: List[np.ndarray], byz_clients) -> np.ndarray:
        # Used for synchronous aggregation
        logging.info(f'Byzantine clients in this round: {byz_clients}')
        logging.info(f"{'='*15}")
        logging.info(f'BYZ PRESENT ? {any([x[1] for x in byz_clients])}')
        logging.info(f"{'='*15}")

        # Step 1: Transform the model weights in model differences (approximations to gradients).
        server_weight_vec = self.get_model_dict_vector()
        # params = [x - server_weight_vec for x in params]
        # Step 2: Perform flame
        logging.info(f'Users: {self.n}, byz: {self.f}')
        args_dict = {'num_users': len(params), 'frac': 1.0, 'malicious': 0.3, 'wrong_mal': 0, 'right_ben': 0, 'turn':0}

        updated_model_vec, accepted = flame_v2(params, self.get_model_dict_vector(), args_dict, 1, use_sync=True,min_cluster_size=self.min_cluster_size)

        self.load_model_dict_vector(updated_model_vec)
        self.incr_age()
        # logging.info(updated_model_vec)
        return updated_model_vec.copy()
        # return super().aggregate_sync(params, byz_clients)

    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> np.ndarray:
        # Used for asynchronous aggregation
        logging.info(f'Flame dict_vector update of client {client_id}')
        # logging.info(f'[Flame Server] weight vector: {weight_vec}')

        if np.isnan(weight_vec).any():
            logging.warning('Client has submitted a vector containing NaN values. Skipping because will crash hdbscan! Returning current server model')
        else:
            server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
            approx_grad = weight_vec - self.model_history[server_model_age] # Gradient approximation

            # @TODO: Adapt the flame algorithm for vectors
            self.model_history_dict[client_id] = (weight_vec, approx_grad)
            self.model_full_history.append([weight_vec, approx_grad])
            # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])
            self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine])
            if len(self.model_full_history) >= max(self.f * 2 + 1, 2):
                local_models = [item[0] for item in self.model_full_history[-self.hist_size:]]
                update_params_list = [item[1] for item in self.model_full_history[-self.hist_size:]]
                args_dict = {'num_users': self.n, 'frac': 1.0, 'malicious': 0.2, 'wrong_mal': 0, 'right_ben': 0, 'turn':0}
                alpha = self.learning_rate / float(max(gradient_age, 1))
                # global_model, accepted = flame(local_models, update_params_list, self.get_model_dict_vector(), args_dict, alpha)

                # @TODO: Current problem: flame expects gradients, currently it gets model weigths?
                logging.info(f'Has Byzantine ? {is_byzantine}')
                clustered_tuples = flame_v3(local_models, self.get_model_dict_vector(), self.min_cluster_size)
                logging.info(f'Has Byzantine ? {is_byzantine}')
                # [logging.info(x) for x in clustered_tuples]
                last_is_valid = clustered_tuples[-1][0]
                if last_is_valid:
                    logging.info(f'Updating server model with alpha: {alpha}')
                    global_model = no_defense_vec_update([clustered_tuples[-1][1]], self.get_model_dict_vector(), alpha)
                    self.load_model_dict_vector(global_model)
                    self.model_history.append(self.get_model_dict_vector())
                    self.incr_age()
                else:
                   self.model_full_history.pop()
                # # logging.info(f'Clustered tuples: {clustered_tuples}')



                # global_model, accepted = flame_v2(local_models, self.get_model_dict_vector(), args_dict, alpha, min_cluster_size=self.min_cluster_size)
                # # print(global_model.shape)
                # self.load_model_dict_vector(global_model)
                # self.model_history.append(self.get_model_dict_vector())
                # self.incr_age()
        return self.get_model_dict_vector().copy()
        # return super().client_weight_dict_vec_update(client_id, weight_vec, gradient_age, is_byzantine)

    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool): 
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
        update_params = get_update(weights, self.model_history[server_model_age])
        client_weight_vec = parameters_dict_to_vector_flt(weights)


        # Save model weights to dict
        self.model_history_dict[client_id] = (weights, update_params)
        self.model_full_history.append([weights, update_params])
        # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])
        self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine])

        if len(self.model_full_history) >= max(self.f * 2 + 1, 2):
            # local_models = [item[0] for key, item in self.model_history_dict.items() if key != client_id] + [weights]
            # update_params_list = [item[1] for key, item in self.model_history_dict.items() if key != client_id] + [update_params]
            # history_size = len(self.model_full_history)
            # history_size = self.f * 2 + 1

            local_models = [item[0] for item in self.model_full_history[-self.hist_size:]]
            update_params_list = [item[1] for item in self.model_full_history[-self.hist_size:]]
            args_dict = {'num_users': self.n, 'frac': 1.0, 'malicious': 0.3, 'wrong_mal': 0, 'right_ben': 0, 'turn':0}
            alpha = self.learning_rate / float(max(gradient_age, 1))
            global_model, accepted = flame(local_models, update_params_list, self.get_model_weights(), args_dict, alpha)
            self.set_weights(global_model)
            # logging.info(f'[{is_byzantine},{accepted == is_byzantine:5}]Server accepts update? {accepted}, is byzantine ? {is_byzantine}')


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