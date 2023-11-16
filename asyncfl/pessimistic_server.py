import logging
from typing import List, Tuple, Union
from asyncfl.flame import flame, flame_v2, flame_v3, flame_v3_aggregate, flame_v3_aggregate_grads, flame_v3_clipbound, flame_v3_filtering
from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt
import math
import numpy as np




class PessimisticServer(Server):

    # Server age is already present in Server
    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)

        self.idle_clients = []
        self.clipbounds = {} # S
        self.k = k
        self.aggregation_bound = aggregation_bound
        if self.aggregation_bound == None:
            # self.aggregation_bound = n
            self.aggregation_bound = 2 * f + 1
        self.aggregation_bound = max(self.aggregation_bound,2)
        try:
            assert self.aggregation_bound > 1
        except Exception as e:
            logging.error(f'Invalid aggregation bound {self.aggregation_bound=} and {self.f=}')
            raise e
        self.pending = {}
        self.processed = {}

        self.disable_alpha = disable_alpha

         
    def remove_oldest_k(self, pending: dict):
        oldest = min(pending.keys())
        # logging.info(f'Oldest is {oldest=}')
        del pending[oldest]
        return pending

    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> np.ndarray:
        # logging.info('Pessimistic server overload')
        logging.info(f'[PessServer] processing client {client_id} with time_delta {self.sched_ctx.current_client_time=}')
        # logging.info(f'{self.sched_ctx.clients_adm}')
        assert self.aggregation_bound != None

        if self.age == gradient_age:
            if self.age not in self.pending:
                self.pending[self.age] = {}
            # pending[a] = pending[a] U client weights
            self.pending[self.age][client_id] = weight_vec

            if len(self.pending[self.age]) >= self.aggregation_bound:
            # if len(self.pending[self.age]) >= self.n:
                # logging.info(f'(Client {client_id}) AGGREGATE!!\t\t  -->\t {len(self.pending[self.age])} >= {self.n}')
                from_k = max(self.age - self.k + 1, 0)
                # to_k = max(self.age - 1, 0)
                to_k = max(self.age, 0) # Change to make it work with range
                W_i = [] # Store delayed aggregate and number of used weights
                for i in range(from_k, to_k-1):
                    if i not in self.pending:
                        self.pending[i] = {}
                    if i not in self.processed:
                        self.processed[i] = {}
                    # logging.info(f'{i=}')
                    assert i in self.pending
                    assert i in self.processed
                    client_weights = list({**self.processed[i], **self.pending[i]}.values())

                    if len(client_weights) < 2:
                        logging.warning('Too little weigths! for delayed aggregation. Skipping for now')
                        continue
                        # logging.error(f'Too little weigths! {i=} {len(client_weights)=}, {len(self.processed[i])=}, {len(self.pending[i])=}')
                        # raise Exception('Too little weigths!')
                        

                    if self.pending[i] != {}:
                        # logging.info(f'{self.pending=}')
                        # logging.info(f'{self.processed=}')
                        # logging.info('Error 1')
                        try:
                            _, euc_dists = flame_v3_clipbound(client_weights)
                            filtered_weights_i,benign_clients = flame_v3_filtering(client_weights, min_cluster_size=max(self.f+1,2))
                            # (many == test).all(axis=1).any(axis=0)
                            filtered_weights_i = [x for x in filtered_weights_i if (list(self.pending[i].values()) == x).all(axis=1).any(axis=0)]
                            # filtered_weights_i = [x for x in filtered_weights_i if any(x in list(self.pending[self.age].values()))]
                            W_i.append((i, len(filtered_weights_i),flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), self.clipbounds[i])))
                        except Exception as e:
                            logging.warning(f'{list(self.pending[i].values())=}')
                            # logging.warning(f'{filtered_weights_i=}')
                            logging.error(e)

                            raise e
                # logging.info('Error 2')
                self.clipbounds[self.age], euc_dists = flame_v3_clipbound(list(self.pending[self.age].values()))
                filtered_weights, benign_clients = flame_v3_filtering(list(self.pending[self.age].values()), min_cluster_size=max(self.f+1,2))
                euc_dists = [x for idx, x in enumerate(euc_dists) if idx in benign_clients]

                # @TODO: Add server learning rate?
                W_hat = flame_v3_aggregate(self.get_model_dict_vector(), filtered_weights, euc_dists, self.clipbounds[self.age])



                grads = []
                current_model = self.get_model_dict_vector()
                updated_model = current_model + ((2* self.f + 1)/ float(self.n))*(W_hat - current_model)
                for grad_age, num_weights, delayed_weights in W_i:
                    # Staleness func?
                    alpha = self.learning_rate / float(max(grad_age, 1))

                    if self.disable_alpha:
                        alpha = 1.0 # Negates the effect of staleness function
                    # num_factor = 
                    # staleness_factor = (float(num_weights) / (float(num_weights) + float(2.0*self.f + 1))) * alpha
                    # W_hat = alpha * num_weights * delayed_grads

                    # @TODO: Add server learning rate --> No, it is incorparated in alpha?
                    # @TODO: Add option to disable alpha?
                    updated_model = updated_model + alpha *(num_weights / float(self.n))* (delayed_weights - self.model_history[grad_age])
 

                self.load_model_dict_vector(updated_model)
                self.model_history.append(updated_model)

                # logging.info(f'Size pending: {len(self.pending)=}, ==> {from_k=}, {to_k=}')
                for i in range(from_k, to_k+1):
                    if i not in self.processed:
                        self.processed[i] = {}
                    self.processed[i] = self.pending[i]
                    self.pending[i] = {}
                    # logging.info(f'Del pending: {i=}')

                # logging.info(f'Size pending 2: {len(self.pending)=}')
                
                self.idle_clients.append(client_id)
                
                for d in self.idle_clients:
                    # age_client_models[d] = self.age + 1 # If server needs to keep track of the client age
                    # Send model to client
                    self.sched_ctx.send_model_to_client(d, self.get_model_dict_vector(), self.age + 1) #type: ignore
                    # logging.info(f'Setting client {d} to compute from pess server')
                    self.sched_ctx.move_client_to_compute_mode(d) #type: ignore 


                self.idle_clients = []
                del_key = self.age - self.k + 1
                if del_key in self.processed:
                    del self.processed[self.age - self.k + 1]
                # @TODO: Delete S[self.age - self.k + 1]
                self.incr_age()
                # logging.info('Choosing branch option 1')

            else:
                # if len(self.pending[self.age]) >= self.aggregation_bound
                # logging.info(f'Waiting for {self.aggregation_bound - len(self.pending[self.age])} more clients')
                # Add to idle clients
                # logging.info(f'Moving client {client_id} to idle')
                self.idle_clients.append(client_id)
                self.sched_ctx.move_client_to_idle_mode(client_id) #type:ignore
                return
        else:
            if self.age - self.k <  gradient_age < self.age - 1:
                # logging.info('Choosing branch option 2A')
                self.pending[gradient_age][client_id] = weight_vec
            # logging.info(f'(Client {client_id}) Choosing branch option 2 because {self.age=} =?= {gradient_age=}')
            
            # Send w to client
            self.sched_ctx.send_model_to_client(client_id, self.get_model_dict_vector(), self.age) #type: ignore
        
        self.sched_ctx.move_client_to_compute_mode(client_id) #type: ignore 


