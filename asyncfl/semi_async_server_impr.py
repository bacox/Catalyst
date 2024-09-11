import logging
from typing import List, Tuple, Union, Dict
from asyncfl.fedAsync_server import fed_async_avg_np
from asyncfl.flame import flame, flame_v2, flame_v3, flame_v3_aggregate, flame_v3_aggregate_grads, flame_v3_clipbound, flame_v3_filtering
from asyncfl.reporting import report_data
from asyncfl.semi_async_server import SemiAsync
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




class SemiAsyncImproved(SemiAsync):

    # Server age is already present in Server
    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005,backdoor_args = {}, project_name = None, aux_meta_data = {}, k: int = 5, aggregation_bound: Union[int, None] = None, disable_alpha: bool = False, reporting=False) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate, backdoor_args, project_name, aux_meta_data, k, aggregation_bound, disable_alpha, reporting)


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
            logging.info(f'[IMPROVED] Fast client {client_id} with 2*cct = {2*cct} <= {self.ert=}')
            fast_client_categorization = 'SF'
        elif 3*cct <= 2*self.ert:
            logging.info(f'[IMPROVED] Fast enough client {client_id} with 3*cct = {3*cct} <= 2*{self.ert=}')
            fast_client_categorization = 'FE'
        else:
            logging.info(f'[IMPROVED] Negligible fast client {client_id} with 3*cct = {3*cct} > 2*{self.ert=}')
            fast_client_categorization = 'NF'

        gt_data = {
            'client_id': client_id,
            'cct': cct,
            'ert': self.ert,
            'categorization': fc_categories[fast_client_categorization],
        }
         
    