import logging
from typing import List, Tuple, Union
from asyncfl.flame import flame, flame_v2, flame_v3, flame_v3_aggregate, flame_v3_aggregate_grads, flame_v3_clipbound, flame_v3_filtering
from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt
import math
import numpy as np




class PessimisticServer(Server):

    # Server age is already present in Server
    # def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False, enable_scaling_factor: bool = True, impact_delayed: float = 1.0) -> None:
    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005, backdoor_args = {}, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False, enable_scaling_factor: bool = True, impact_delayed: float = 1.0) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate, backdoor_args)

        self.idle_clients = []
        self.clipbounds = {} # S
        self.k = k
        self.aggregation_bound = aggregation_bound
        
        if self.aggregation_bound == None:
            self.aggregation_bound = 2 * f + 1
        self.aggregation_bound = max(self.aggregation_bound,2)
        self.f = max(self.f, 2)
        
        try:
            assert self.aggregation_bound > 1
        except Exception as e:
            logging.error(f'Invalid aggregation bound {self.aggregation_bound=} and {self.f=}')
            raise e
        self.pending = {}
        self.processed = {}
        self.disable_alpha = disable_alpha
        self.enable_scaling_factor = enable_scaling_factor
        self.impact_delayed = impact_delayed
        self.model_history: list = [None] * self.k
        self.enable_scaling_factor = enable_scaling_factor
        self.impact_delayed = impact_delayed
    
    def remove_oldest_k(self, pending: dict):
        oldest = min(pending.keys())
        del pending[oldest]
        return pending

    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[np.ndarray, bool]:
        has_aggregated = False
        logging.info(f'[PessServer] processing client {client_id}')
        # logging.info(f'[PessServer] processing client {client_id} with time_delta {self.sched_ctx.current_client_time=}')
        assert self.aggregation_bound != None

        # Check what client we are dealing with
        if self.age == gradient_age:
            if self.age not in self.pending:
                self.pending[self.age] = {}
            # pending[a] = pending[a] U client weights
            self.pending[self.age][client_id] = weight_vec

            if len(self.pending[self.age]) >= self.aggregation_bound:
                # logging.info('[PessServer] Aggregate?')
                from_k = max(self.age - self.k + 1, 0)
                to_k = max(self.age, 0) # Change to make it work with range
                W_i = [] # Store delayed aggregate and number of used weights

                # Loop over older delayed updates
                for i in range(from_k, to_k-1):
                    if i not in self.pending:
                        self.pending[i] = {}
                    if i not in self.processed:
                        self.processed[i] = {}
                    assert i in self.pending
                    assert i in self.processed
                    client_weights = list({**self.processed[i], **self.pending[i]}.values())
                    if len(client_weights) < 2:
                        logging.warning('Too little weigths! for delayed aggregation. Skipping for now')
                        continue

                    if self.pending[i] != {}:
                        try:
                            _, euc_dists = flame_v3_clipbound(client_weights)
                            filtered_weights_i,benign_clients = flame_v3_filtering(client_weights, min_cluster_size=max(self.f + 1, 2))
                            filtered_weights_i = [x for x in filtered_weights_i if (list(self.pending[i].values()) == x).all(axis=1).any(axis=0)]
                            W_i.append((i, len(filtered_weights_i),flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), self.clipbounds[i])))
                        except Exception as e:
                            logging.warning(f'{list(self.pending[i].values())=}')
                            logging.error(e)
                            raise e
                        
                self.clipbounds[self.age], euc_dists = flame_v3_clipbound(list(self.pending[self.age].values()))
                filtered_weights, benign_clients = flame_v3_filtering(list(self.pending[self.age].values()), min_cluster_size=max(self.f + 1, 2))
                euc_dists = [x for idx, x in enumerate(euc_dists) if idx in benign_clients]
                # @TODO: Add server learning rate?
                W_hat = flame_v3_aggregate(self.get_model_dict_vector(), filtered_weights, euc_dists, self.clipbounds[self.age])
                grads = []
                current_model = self.get_model_dict_vector()
                # logging.info(f'[PessServer] Aggregate! with ratio {(2* self.f + 1)/ float(self.n)}')

                # Old update function
                # updated_model = current_model + ((2* self.f + 1)/ float(self.n))*(W_hat - current_model)

                # New update function
                scaling_factor = 1
                if self.enable_scaling_factor:
                    scaling_factor = (len(filtered_weights)/self.n)
                updated_model = (1-scaling_factor)*current_model + scaling_factor* W_hat

                for grad_age, num_weights, delayed_weights in W_i:
                    # Staleness func?
                    alpha = self.learning_rate / float(max(grad_age, 1))
                    if self.disable_alpha:
                        alpha = 1.0 # Negates the effect of staleness function
                    # updated_model = updated_model + self.impact_delayed * alpha *(num_weights / float(self.n))* (delayed_weights - self.model_history[grad_age])
                    updated_model = updated_model + self.impact_delayed * alpha *(num_weights / float(self.n))* (delayed_weights - self.model_history[grad_age % self.k])
 
                self.load_model_dict_vector(updated_model)
                # self.model_history.append(updated_model)
                has_aggregated = True
                self.model_history[self.age % self.k] = updated_model

                # Clear used values
                for i in range(from_k, to_k+1):
                    if i not in self.processed:
                        self.processed[i] = {}
                    self.processed[i] = self.pending[i]
                    self.pending[i] = {}

                # self.idle_clients.append(client_id)
                # logging.debug('Pre compute')
                for d in self.idle_clients:
                    # Send model to client
                    # logging.debug(f'[Pess] Moving client {d} to compute')
                    self.sched_ctx.send_model_to_client(d, self.get_model_dict_vector(), self.age + 1) #type: ignore
                    self.sched_ctx.move_client_to_compute_mode(d) #type: ignore 

                self.idle_clients = []
                del_key = self.age - self.k + 1
                if del_key in self.processed:
                    del self.processed[self.age - self.k + 1]
                # @TODO: Delete S[self.age - self.k + 1]
                self.incr_age()
            else:
                # Add to idle clients
                self.idle_clients.append(client_id)
                self.sched_ctx.move_client_to_idle_mode(client_id) #type:ignore
                return None, has_aggregated
        else:
            if self.age - self.k <  gradient_age < self.age - 1:
                self.pending[gradient_age][client_id] = weight_vec            
            # Send w to client
            self.sched_ctx.send_model_to_client(client_id, self.get_model_dict_vector(), self.age) #type: ignore
        # logging.debug(f'Last statement: moving client {client_id} to compute')
        self.sched_ctx.move_client_to_compute_mode(client_id) #type: ignore 
        return None, has_aggregated


