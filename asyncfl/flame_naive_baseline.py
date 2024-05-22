import logging
from typing import List, Tuple, Union
from asyncfl.flame import flame, flame_v2, flame_v3, flame_v3_aggregate, flame_v3_aggregate_grads, flame_v3_clipbound, flame_v3_filtering
from asyncfl.reporting import report_data
from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt
import math
import numpy as np




class FlameNaiveBaseline(Server):

    # Server age is already present in Server
    # def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False, enable_scaling_factor: bool = True, impact_delayed: float = 1.0) -> None:
    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005, backdoor_args = {}, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False, enable_scaling_factor: bool = True, impact_delayed: float = 1.0, alg_version: str = 'A',project_name = None, aux_meta_data={}, mitigate_staleness = True, reporting = True) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate, backdoor_args, project_name=project_name, aux_meta_data=aux_meta_data, reporting=reporting)

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
        self.pending_weights: List[np.ndarray] = []
        self.byz_client_hist : List[bool] = []
        self.pending = {}
        self.processed = {}
        self.disable_alpha = disable_alpha
        self.enable_scaling_factor = enable_scaling_factor
        self.impact_delayed = impact_delayed
        self.model_history: list = [None] * self.k
        self.enable_scaling_factor = enable_scaling_factor
        self.impact_delayed = impact_delayed
        self.alg_version = alg_version
        self.discarded_updates = 0
        self.discarded_ptr = 0
    
    def remove_oldest_k(self, pending: dict):
        oldest = min(pending.keys())
        del pending[oldest]
        return pending

    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[Union[np.ndarray, None], bool]:
        logging.info(f'[NaiveFlame {self.alg_version}] processing client {client_id} :: {is_byzantine=} :: {gradient_age=} :: {self.age=}')
        if self.alg_version == 'A':
            return self.weight_update_A(client_id, weight_vec, gradient_age, is_byzantine)
        elif self.alg_version == 'B':
            return self.weight_update_B(client_id, weight_vec, gradient_age, is_byzantine)
        elif self.alg_version == 'B2':
            return self.weight_update_B2(client_id, weight_vec, gradient_age, is_byzantine)
        elif self.alg_version == 'C':
            return self.weight_update_C(client_id, weight_vec, gradient_age, is_byzantine)
        else:
            raise Exception(f'Unknown algorithm version "{self.alg_version}"')
    
    def weight_update_A(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[np.ndarray, bool]:
        '''
        This version of Flame just waits for 2f+1 updates to arrive.
        It does not matter how many times a clients updates, the model aggregates after 2f+1.
        @TODO: Keep track of the number of pending weights and make sure that they are removed after aggregation?
        '''
        has_aggregated = False
        assert self.aggregation_bound != None
        self.pending_weights.append(weight_vec)
        self.byz_client_hist.append(is_byzantine)

        if len(self.pending_weights) >= self.aggregation_bound:
            client_weights = self.pending_weights[-self.aggregation_bound:]
            frac_byz = sum(self.byz_client_hist[-self.aggregation_bound:]) / float(len(self.byz_client_hist[-self.aggregation_bound:]))
            logging.info(f'[NaiveFlame {self.alg_version}] {len(client_weights)=}, percentage byzantine: {frac_byz}')
            # Do flame stuff
            clip_value, euc_dists = flame_v3_clipbound(client_weights)
            filtered_weights_i, benign_clients = flame_v3_filtering(client_weights, min_cluster_size=max(self.f + 1, 2))
            updated_weights = flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), clip_value)
            self.load_model_dict_vector(updated_weights)
            has_aggregated = True
            self.pending_weights = []
            self.age += 1
        report_data(self.wandb_obj, {'pending_updates': len(self.pending_weights), 'alg_version': self.alg_version})
        return self.get_model_dict_vector(), has_aggregated

    def weight_update_B(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[np.ndarray, bool]:
        '''
        Same as version A but the server wait for 2f+1 clients to send an update.
        Clients continue to train locally after sending an update. They can improve the update; a dictionary keeps track of the clients. Newer updates overwrite old ones.
        '''
        
        has_aggregated = False
        assert self.aggregation_bound != None
        self.byz_client_hist.append(is_byzantine)
        self.pending[client_id] = weight_vec
        # self.pending_weights.append(weight_vec)
        

        if len(self.pending) >= self.aggregation_bound:
            client_weights = list(self.pending.values())
            # frac_byz = sum(self.byz_client_hist[-self.aggregation_bound:]) / float(len(self.byz_client_hist[-self.aggregation_bound:]))
            frac_byz = '?'
            logging.info(f'[NaiveFlame {self.alg_version}] {len(client_weights)=}, percentage byzantine: {frac_byz}')
            # Do flame stuff
            clip_value, euc_dists = flame_v3_clipbound(client_weights)
            filtered_weights_i, benign_clients = flame_v3_filtering(client_weights, min_cluster_size=max(self.f + 1, 2))
            updated_weights = flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), clip_value)
            self.load_model_dict_vector(updated_weights)
            has_aggregated = True
            self.pending = {}
            self.age += 1
        report_data(self.wandb_obj, {'pending_updates': len(self.pending), 'alg_version': self.alg_version})
        return self.get_model_dict_vector(), has_aggregated


    def weight_update_B2(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[np.ndarray, bool]:
        '''
        Same as version B but old updates are discarted. The age of update must be equal to the current server model age.
        '''
        has_aggregated = False
        assert self.aggregation_bound != None
        # self.pending_weights.append(weight_vec)

        if self.age == gradient_age:
            self.byz_client_hist.append(is_byzantine)
            self.pending[client_id] = weight_vec
        else:
            logging.info(f'[NaiveFlame {self.alg_version}] Discarding update from {client_id} because of age mismatch. {gradient_age=}, {self.age=}')
            self.discarded_updates += 1
            return self.get_model_dict_vector(), has_aggregated

        unique_discarded = 0
        if len(self.pending) >= self.aggregation_bound:
            logging.info(f'[NaiveFlame] <!!> Start Aggregation {self.age=}')
            client_weights = list(self.pending.values())
            client_ids = list(self.pending.keys())
            # frac_byz = sum(self.byz_client_hist[-self.aggregation_bound:]) / float(len(self.byz_client_hist[-self.aggregation_bound:]))
            frac_byz = '?'
            logging.info(f'[NaiveFlame {self.alg_version}] {len(client_weights)=}, percentage byzantine: {frac_byz}')
            # Do flame stuff
            clip_value, euc_dists = flame_v3_clipbound(client_weights)
            filtered_weights_i, benign_clients = flame_v3_filtering(client_weights, min_cluster_size=max(self.f + 1, 2))
            updated_weights = flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), clip_value)
            self.load_model_dict_vector(updated_weights)
            has_aggregated = True
            self.pending = {}
            self.age += 1
            discarded_updates_since_last_aggregation = self.discarded_updates - self.discarded_ptr
            unique_discarded = self.n - len(client_ids)
            logging.info(f'[NaiveFlame {self.alg_version}] Aggregated {len(client_weights)=}, percentage byzantine: {frac_byz}, {unique_discarded=}, {discarded_updates_since_last_aggregation=}')
            
        else:
            logging.info(f'[NaiveFlame {self.alg_version}] Waiting for more updates {len(self.pending)=} / {self.aggregation_bound}')
        if has_aggregated:
            
            report_data(self.wandb_obj, {'pending_updates': len(self.pending), 'alg_version': self.alg_version, 'num_discarded': self.discarded_updates, 'unique_discarded': unique_discarded, 'used_clients': len(client_ids), 'discarted_for_agg': discarded_updates_since_last_aggregation, 'used_clip_value': clip_value})
            self.discarded_ptr = self.discarded_updates
        else:
            report_data(self.wandb_obj, {'pending_updates': len(self.pending), 'alg_version': self.alg_version, 'num_discarded': self.discarded_updates})
        
        return self.get_model_dict_vector(), has_aggregated

    def weight_update_C(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[Union[np.ndarray, None], bool]:
        '''
        Same as version B but clients who have sent an update remain idle until the aggregation has happened.
        '''
        
        has_aggregated = False
        assert self.aggregation_bound != None

        self.pending[client_id] = weight_vec
        # self.pending_weights.append(weight_vec)
        if len(self.pending) >= self.aggregation_bound:
            client_weights = list(self.pending.values())
            # frac_byz = sum(self.byz_client_hist[-self.aggregation_bound:]) / float(len(self.byz_client_hist[-self.aggregation_bound:]))
            frac_byz = '?'
            logging.info(f'[NaiveFlame {self.alg_version}] {len(client_weights)=}, percentage byzantine: {frac_byz}')
            # Do flame stuff
            clip_value, euc_dists = flame_v3_clipbound(client_weights)
            filtered_weights_i, benign_clients = flame_v3_filtering(client_weights, min_cluster_size=max(self.f + 1, 2))
            updated_weights = flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), clip_value)
            self.load_model_dict_vector(updated_weights)
            has_aggregated = True
            self.pending = {}
            # Update waiting clients
            for d in self.idle_clients:
                # Send model to client
                self.sched_ctx.send_model_to_client(d, self.get_model_dict_vector(), self.age + 1) #type: ignore
                self.sched_ctx.move_client_to_compute_mode(d) #type: ignore 
            report_data(self.wandb_obj, {'pending_updates': len(self.pending), 'alg_version': self.alg_version})
            self.age += 1
            return self.get_model_dict_vector(), has_aggregated
        
        else:
            # Add to idle clients
            self.idle_clients.append(client_id)
            self.sched_ctx.move_client_to_idle_mode(client_id) #type:ignore
            report_data(self.wandb_obj, {'pending_updates': len(self.pending), 'alg_version': self.alg_version})
            return None, has_aggregated