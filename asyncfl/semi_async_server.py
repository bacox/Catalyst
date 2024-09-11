import logging
from typing import List, Tuple, Union, Dict
from asyncfl.fedAsync_server import fed_async_avg_np
from asyncfl.flame import flame, flame_v2, flame_v3, flame_v3_aggregate, flame_v3_aggregate_grads, flame_v3_clipbound, flame_v3_filtering
from asyncfl.reporting import report_data
from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt
from dataclasses import dataclass, field
import math
import numpy as np

@dataclass
class Bucket:
    models: List = field(default_factory=lambda: [])

    def add_model(self, model):
        self.models.append(model)

    def get_next_model(self):
        return self.models[-1]

@dataclass
class Buckets():
    buckets: Dict[int, Bucket] = field(default_factory=lambda: {})

    def get_bucket(self, age: int, default_model):
        if not age in self.buckets:
            self.buckets[age] = Bucket()
            self.buckets[age].add_model(default_model)
        return self.buckets[age]




class SemiAsync(Server):

    # Server age is already present in Server
    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005,backdoor_args = {}, project_name = None, aux_meta_data = {}, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False, reporting=False) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate, backdoor_args, project_name=project_name, aux_meta_data=aux_meta_data, reporting=reporting)

        self.idle_clients = []
        self.clipbounds = {} # S
        self.k = k
        self.aggregation_bound = aggregation_bound


        self.lat = 0 # Last aggregation timestamp
        # Set estimated round time to infinity
        self.ert = math.inf

        self.client_start_timestamps = {i: 0 for i in range(self.n)}

        logging.info(f' self.client_start_timestamps= {self.client_start_timestamps=}')


        if self.aggregation_bound == None:
            # self.aggregation_bound = n
            self.aggregation_bound = 2 * f + 1
        self.aggregation_bound = max(self.aggregation_bound,2)
        try:
            assert self.aggregation_bound > 1
        except Exception as e:
            logging.error(f'Invalid aggregation bound {self.aggregation_bound=} and {self.f=}')
            raise e
        # self.aggregation_bound = n
        logging.info(f'[Semi] Aggregation bound {self.aggregation_bound=} with {f=}')
        self.pending = {}
        self.processed = {}

        self.disable_alpha = disable_alpha
        self.idle_clients_count = []

        # To keep track of the fast clients
        self.fast_clients: List[int] = []
        self.buckets = Buckets()



         
    def remove_oldest_k(self, pending: dict):
        oldest = min(pending.keys())
        # logging.info(f'Oldest is {oldest=}')
        del pending[oldest]
        return pending
    
    # def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool)-> Tuple[Union[np.ndarray, None], bool]:
    #     assert self.aggregation_bound != None
    #     has_aggregated = False

    #     if self.age == gradient_age: # If this is an update for the round
            

    #         pass
    #     else:
    #         if self.age - self.k <  gradient_age < self.age - 1:
    #             # self.pending[gradient_age][client_id] = weight_vec   
    #             self.pending[gradient_age][client_id] = (weight_vec, is_byzantine)         
    #         # Send w to client
    #         self.sched_ctx.send_model_to_client(client_id, self.get_model_dict_vector(), self.age) #type: ignore
    #     # logging.debug(f'Last statement: moving client {client_id} to compute')
    #     self.sched_ctx.move_client_to_compute_mode(client_id) #type: ignore 
    #     return None, has_aggregated

    def handle_slow_clients(self, gradient_age: int, client_id: int, weight_vec: np.ndarray, is_byzantine: bool):
        if self.age - self.k <  gradient_age < self.age - 1:
            # self.pending[gradient_age][client_id] = weight_vec   
            self.pending[gradient_age][client_id] = (weight_vec, is_byzantine)         
        # Send w to client
        self.sched_ctx.send_model_to_client(client_id, self.get_model_dict_vector(), self.age) #type: ignore

    def handle_fast_clients(self, client_id: int, weight_vec: np.ndarray, cct: float):
        """_summary_
        Args:
            client_id (int): client id
            weight_vec (np.ndarray): model weights
            cct (float): Client computing time
           
        """

        # Fast client categorization
        # SF: Fast client 2*cct <= ert
        # FE: Fast enough client 3*cct <= 2*ert
        # NF: Negligible fast client is the rest

        fast_client_categorization = 'NF'
        fc_categories = {
            'NF': 1,
            'SF': 2,
            'FE': 3
        }

        if 2*cct <= self.ert:
            logging.info(f'[Semi] Fast client {client_id} with 2*cct = {2*cct} <= {self.ert=}')
            fast_client_categorization = 'SF'
        elif 3*cct <= 2*self.ert:
            logging.info(f'[Semi] Fast enough client {client_id} with 3*cct = {3*cct} <= 2*{self.ert=}')
            fast_client_categorization = 'FE'
        else:
            logging.info(f'[Semi] Negligible fast client {client_id} with 3*cct = {3*cct} > 2*{self.ert=}')
            fast_client_categorization = 'NF'

        gt_data = {
            'client_id': client_id,
            'cct': cct,
            'ert': self.ert,
            'categorization': fc_categories[fast_client_categorization],
        }

        report_data(self.wandb_obj, gt_data)
        # Get bucket for client
        bucket = self.buckets.get_bucket(self.age, self.get_model_dict_vector())
        current_bucket_model = bucket.get_next_model()
        # logging.info(f'[Semi] Fast client {client_id} get current model {current_bucket_model=}')
        self.personalized_update(client_id, weight_vec, current_bucket_model)
        self.fast_clients.append(client_id)

    def personalized_update(self, client_id: int, weight_vec: np.ndarray, current_server_model: np.ndarray):
        alpha_averaged: np.ndarray = fed_async_avg_np(weight_vec, current_server_model, self.learning_rate)
        # logging.info(f'[Semi] Fast client {client_id} get alpha-averaged model {alpha_averaged=}')
        self.sched_ctx.send_model_to_client(client_id, alpha_averaged, self.age)
        self.sched_ctx.move_client_to_compute_mode(client_id, partial=True)

    def compute_delayed_aggregation(self, from_k: int, to_k: int) -> List[Tuple[int, int, np.ndarray]]:
        W_i = [] # Store delayed aggregate and number of used weights
        for i in range(from_k, to_k-1):
            if i not in self.pending:
                self.pending[i] = {}
            if i not in self.processed:
                self.processed[i] = {}
            assert i in self.pending
            assert i in self.processed
            client_weights = list({**self.processed[i], **self.pending[i]}.values())


            kth_age_values = list([x[0] for x in client_weights])
            kth_age_byzantines = [x[1] for x in client_weights]
            logging.info(f'[Semi] kth_age_values= {kth_age_values=}')
            logging.info(f'[Semi] client_weights_values= {client_weights=}')
            # logging.info(f'[Semi] current_age_byzantines= {current_age_byzantines=}')
            
            if len(client_weights) < 2:
                logging.warning('Too little weigths! for delayed aggregation. Skipping for now')
                continue

            if self.pending[i] != {}:
                try:
                    # Log the types of kth_age_values
                    logging.info(f'[Semi] kth_age_values= {kth_age_values=}')
                    _, euc_dists = flame_v3_clipbound(kth_age_values)

                    pending_i_values = list([x[0] for x in self.pending[i].values()])
                    _pending_i_byzantines = [x[1] for x in self.pending[i].values()]
                    filtered_weights_i,benign_clients = flame_v3_filtering(kth_age_values, min_cluster_size=max(self.f+1,2))
                    filtered_weights_i = [x for x in filtered_weights_i if (pending_i_values == x).all(axis=1).any(axis=0)]
                    W_i.append((i, len(filtered_weights_i),flame_v3_aggregate(self.get_model_dict_vector(),filtered_weights_i, euc_dists.tolist(), self.clipbounds[i])))
                except Exception as e:
                    logging.warning(f'{pending_i_values=}')
                    logging.error(e)
                    raise e
        return W_i

    def process_partial_epochs(self):
        # Recal fast clients?
        logging.debug(f'Recalling fast clients: {list(set(self.fast_clients))}')
        # logging.debug(f'{self.sched_ctx.clients_adm["computing"]}')
        logging.info(f'Scheduled clients: {self.sched_ctx.clients_adm["computing"]}')
        # Let fast clients finish their partial epochs
        for cc in [x for x in self.sched_ctx.clients_adm['computing'] if x[2]]: # if x[2] is True, then it is a partial computation
            proportional_train_time = min((self.sched_ctx.wall_time - cc[3])/ cc[0], 1)
            self.sched_ctx.client_partial_training(cc[2], proportional_train_time)
            # client_i: Client = 
            self.sched_ctx.move_client_to_idle_mode(cc[2]) #type:ignore


    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[Union[np.ndarray, None], bool]:
        assert self.aggregation_bound != None
        has_aggregated = False
        prefix = ''
        if client_id in self.fast_clients:
            prefix = ' FS'
        
        # Calculate the client computing time
        cct = self.sched_ctx.wall_time - self.client_start_timestamps[client_id]
        self.client_start_timestamps[client_id] = self.sched_ctx.wall_time
        logging.info(f'{prefix} [Semi] Client {client_id} has been computing for {cct} seconds')

        # Check what client we are dealing with
        # If client from the current round?
        if self.age == gradient_age:
            if self.age not in self.pending:
                self.pending[self.age] = {}
            # pending[a] = pending[a] U client weights
            self.pending[self.age][client_id] = (weight_vec, is_byzantine)

            # Check if we have enough updates to aggregate
            if len(self.pending[self.age]) >= self.aggregation_bound:

                # Get current wall time
                current_wall_time = self.sched_ctx.wall_time
                self.ert = current_wall_time - self.lat
                self.lat = current_wall_time
                logging.info(f'[Semi] Aggregate at all times: {current_wall_time=} {self.ert=}')

                # logging.info('[Semi] Aggregate?')
                from_k = max(self.age - self.k + 1, 0)
                to_k = max(self.age, 0) # Change to make it work with range
                
                # Recal fast clients?
                self.process_partial_epochs()
                # Include older updates
                # Use the window [from_k, to_k-1] to look back for pending old updates
                W_i = self.compute_delayed_aggregation(from_k, to_k)

                # Log the client_ids that are used in self.pending[self.age]
                used_client_ids = list(self.pending[self.age].keys())
                logging.info(f'[Semi] used_client_ids= {used_client_ids=}')

                current_age_values = list([x[0] for x in self.pending[self.age].values()])
                current_age_byzantines = [x[1] for x in self.pending[self.age].values()]

                # Log byzantine client_ids
                byzantine_client_ids = [used_client_ids[i] for i in range(len(used_client_ids)) if current_age_byzantines[i]]
                # Get benign clients_ids
                benign_client_ids = [used_client_ids[i] for i in range(len(used_client_ids)) if not current_age_byzantines[i]]

                ground_truth = {
                    'byzantine': byzantine_client_ids,
                    'benign': benign_client_ids
                }
 
                logging.info(f'[Semi] byzantine_client_ids= {byzantine_client_ids=}')
                logging.info(f'[Semi] current_age_values= {current_age_values=}')
                logging.info(f'[Semi] current_age_byzantines= {current_age_byzantines=}')
                self.clipbounds[self.age], euc_dists = flame_v3_clipbound(current_age_values)
                filtered_weights, benign_clients = flame_v3_filtering(current_age_values, min_cluster_size=max(self.f+1,2))

                # Calculate the client_ids that are used in the aggregation
                accepted_client_ids = [used_client_ids[i] for i in range(len(used_client_ids)) if i in benign_clients]
                rejected_client_ids = [used_client_ids[i] for i in range(len(used_client_ids)) if i not in benign_clients]

                # Save the used client_ids and rejected client_ids in the ground truth
                ground_truth['used'] = accepted_client_ids
                ground_truth['rejected'] = rejected_client_ids

                gt_data = {
                    'num_byzantine': len(byzantine_client_ids),
                    'num_benign': len(benign_client_ids),
                    'num_used': len(accepted_client_ids),
                    'num_rejected': len(rejected_client_ids),
                    'age': self.age,
                    'num_idle': len(self.idle_clients),
                    'num_fast': len(self.fast_clients),
                    'num_pending': len(self.pending[self.age]),
                    'n_computing': len(self.sched_ctx.clients_adm['computing']),
                    'n_idle': len(self.sched_ctx.clients_adm['idle']),
                    'ert': self.ert


                }
                report_data(self.wandb_obj, gt_data)

                # Log the ground truth
                logging.info(f'[Semi] ground_truth= {ground_truth=}')
                logging.info(f'[Semi] used_client_ids= {accepted_client_ids=}')

                # log the benign clients
                logging.info(f'[Semi BC] Benign clients: {benign_clients=}, {current_age_byzantines=}')
                euc_dists = [x for idx, x in enumerate(euc_dists) if idx in benign_clients]
                # @TODO: Add server learning rate?
                W_hat = flame_v3_aggregate(self.get_model_dict_vector(), filtered_weights, euc_dists, self.clipbounds[self.age])
                grads = []
                current_model = self.get_model_dict_vector()

                # @TODO: Be consistent when using self.f and self.aggregation_bound!
                # @TODO: How to express the impact of the main flame update and the late updates

                updated_model = W_hat
                # updated_model = current_model + ((self.aggregation_bound)/ float(self.n))*(W_hat - current_model)
                for grad_age, num_weights, delayed_weights in W_i:
                    # Staleness func?
                    alpha = self.learning_rate / float(max(grad_age, 1))
                    if self.disable_alpha:
                        alpha = 1.0 # Negates the effect of staleness function
                    # @TODO: Add server learning rate --> No, it is incorparated in alpha?
                    # @TODO: Add option to disable alpha?
                    updated_model = updated_model + alpha *(num_weights / float(self.n))* (delayed_weights - self.model_history[grad_age])
 
                self.load_model_dict_vector(updated_model)
                self.model_history.append(updated_model)
                has_aggregated = True

                # Clear used values
                for i in range(from_k, to_k+1):
                    if i not in self.processed:
                        self.processed[i] = {}
                    self.processed[i] = self.pending[i]
                    self.pending[i] = {}

                # self.idle_clients.append(client_id)
                # logging.debug('Pre compute')
                logging.info(f'Size of idle clients pre: {len(self.idle_clients)}')
                for d in self.idle_clients:
                    # Send model to client
                    # logging.debug(f'[Pess] Moving client {d} to compute')
                    self.sched_ctx.send_model_to_client(d, self.get_model_dict_vector(), self.age + 1) #type: ignore
                    self.sched_ctx.move_client_to_compute_mode(d) #type: ignore 

                self.idle_clients = []
                logging.info(f'Size of idle clients post: {len(self.idle_clients)}')
                del_key = self.age - self.k + 1
                if del_key in self.processed:
                    del self.processed[self.age - self.k + 1]
                # @TODO: Delete S[self.age - self.k + 1]
                self.incr_age()
            else:
                # This is a fast client

                self.handle_fast_clients(client_id, weight_vec, cct)
                return None, has_aggregated

                # # Add to idle clients
                # self.idle_clients.append(client_id)
                # self.sched_ctx.move_client_to_idle_mode(client_id) #type:ignore
                # return None, has_aggregated
        else:
            self.handle_slow_clients(gradient_age, client_id, weight_vec, is_byzantine)
        # logging.debug(f'Last statement: moving client {client_id} to compute')
        self.sched_ctx.move_client_to_compute_mode(client_id) #type: ignore 
        return None, has_aggregated

