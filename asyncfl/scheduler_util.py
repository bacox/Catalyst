import logging
from typing import List, Union
import numpy as np
from asyncfl.client import Client


class SchedulerContext():
    def __init__(self, clients : List[Client], compute_times: dict) -> None:
        self.clients_adm = {
            'idle': [],
            'computing': []
        }
        self.compute_times = compute_times
        self.clients = clients
        self.current_client_time = 0
        self.wall_time = 0

    def next_client(self):
        # Next client
        # logging.debug(f'{len(self.clients_adm["computing"])=}')
        self.clients_adm['computing'].sort(key=lambda x: x[0])
        # logging.info(f'[SCTX] {self.clients_adm["computing"]}')

        # logging.info(f'{computing_clients=}')
        # logging.info(f"{len(self.clients_adm['computing'])=} && {len(self.clients_adm['idle'])=}")
        client_time, next_client, _partial, _insert_wall_time = self.clients_adm['computing'].pop(0)
        # logging.info(f'[SchedCTX] Next time delta {client_time=}')
        self.current_client_time = client_time
        assert client_time >= 0
        return client_time, next_client
    

    def adjust_time(self, client_time: Union[float, None] = None):
        
        time_delta = self.current_client_time
        if client_time is not None:
            time_delta = client_time
        # logging.info(f'[SchedCTX] Shifting time with {client_time=}')

        for cc in self.clients_adm['computing']:
            try:
                assert cc[0] >= time_delta
            except Exception as e:
                logging.warning(f'[SchedCTX] time shift {cc[0]} >= {time_delta}')
                raise e
            cc[0] -= time_delta
        self.wall_time += time_delta


    def send_model_to_client(self, client_id: int, model_vec: np.ndarray, model_age: int):
        c = next((x for x in self.clients if x.pid == client_id), None)
        assert c is not None
        c.load_model_dict_vector(model_vec)
        c.local_age = model_age

    def move_client_to_idle_mode(self, client_id: int):
        c = next((x for x in self.clients if x.pid == client_id), None)
        assert c is not None
        # logging.debug([x[2] for x in self.clients_adm['computing']])
        self.clients_adm['computing'] = [x for x in self.clients_adm['computing'] if x[1].pid != client_id]
        assert c.pid not in [y.pid for _x, y, z, zz in self.clients_adm["computing"]]
        self.clients_adm['idle'].append(c)

    def move_client_to_compute_mode(self, client_id: int, partial = False):
        # This causes problems because the client is re-inserted into the computing part before adjusting time.
        # @TODO: Make sure this doesn't mess up the adjust time part.
        c = next((x for x in self.clients if x.pid == client_id), None)
        assert c is not None
        self.clients_adm['idle'] = [x for x in self.clients_adm['idle'] if x.pid != c.pid]
        assert c.pid not in [y.pid for y in self.clients_adm["idle"]]

        # Give the client double the compute time because time adjustment happens after this
        self.clients_adm['computing'].append([self.compute_times[c.pid] + self.current_client_time, c, partial, self.wall_time])

    def client_partial_training(self, client_id: int, fraction: float):
        c = next((x for x in self.clients if x.pid == client_id), None)
        assert c is not None
        c.move_to_gpu()
        batch_size = c.train_set.batch_size
        assert batch_size is not None
        batch_limit = int(batch_size / fraction)
        logging.debug(f'Running partial training with {fraction=} and {batch_limit=}')
        c.train(num_batches=batch_limit)
        is_byzantine = c.is_byzantine

        # @TODO: Do something with the data
        data = c.get_model_dict_vector(),c.local_age, is_byzantine
        c.move_to_cpu()