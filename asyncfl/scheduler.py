import asyncio
import copy
import inspect
import json
import logging
import multiprocessing
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Lock, Pool, Process, RLock, current_process
from multiprocessing.pool import AsyncResult
from pathlib import Path
from threading import Thread
from typing import Any, List, Tuple, Union
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb
from wandb import AlertLevel
# os.environ["WANDB_SILENT"] = "true"

from asyncfl.reporting import finish_exp, finish_reporting
# import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

from asyncfl.dataloader import afl_dataset2
from asyncfl.network import flatten
from asyncfl.scheduler_util import SchedulerContext

from .client import Client
from .kardam import Kardam
from .network import flatten, unflatten, unflatten_b
from .server import Server, fed_avg, fed_avg_vec
from .task import Task


def dict_convert_class_to_strings(dictionary: dict):
    d = copy.deepcopy(dictionary)
    d["clients"]["client"] = d["clients"]["client"].__name__
    d["clients"]["f_type"] = d["clients"]["f_type"].__name__
    d["server"] = d["server"].__name__
    # Make sure any numpy arrays are converted to lists
    def dict_walk(data):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = dict_walk(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()
                # print l
        return data
    d = dict_walk(d)
    return d

def wrap_name(name: str, l: int = 13) -> str:
    if len(name) > l:
        return f'{name[:l-6]}...{name[-3:]}'
    return name

class PoolManager:
    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None, context=None) -> None:
        self.pool = Pool(processes, initializer, initargs, maxtasksperchild)
        self.tasks = []
        self.active_tasks = []
        self.results = []
        self.num_processes = processes
        self.capacity = 1.0

    def add_task(self, func, args, required_capacity: float = 0.0):
        self.tasks.append((func, args, required_capacity))

    def run(self, pbar=None):
        while len(self.tasks) or len(self.active_tasks) > 0:
            if self.tasks and self.capacity >= self.tasks[-1][2]:
                func, args, cap = self.tasks.pop()
                self.capacity -= cap
                # print(f'New Cap {self.capacity}')
                self.active_tasks.append((self.pool.apply_async(func, args), cap))

            for index, (t, cap) in enumerate(self.active_tasks):
                if t.ready():
                    self.results.append(t)
                    self.active_tasks.pop(index)
                    self.capacity += cap
                    # print(f'Cap restore {self.capacity}')
                    if pbar:
                        pbar.update(1)

    def get_results(self) -> List[AsyncResult]:
        return self.results


class Scheduler:
    def __init__(self, dataset_name: str, model_name: str, worker_id = 0, project='async-default', exp_name=None, **config):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.clients: List[Client] = []
        self.entities = {}
        self.compute_times = {}
        self.project = project
        self.exp_name = exp_name
        self.aux_meta_data = {
            'exp_name': exp_name
        }

        self.aux_meta_data = {**self.aux_meta_data, **config.get('meta_data', {})}

        self.worker_id = worker_id
        if "learning_rate" not in config["server_args"]:
            config["server_args"]["learning_rate"] = 0.005
        self.create_entities(**config)
        self.dataset = None
        self.metric = "Perplexity" if self.dataset_name == "wikitext2" else "Accuracy"

    def create_entities(self, clients, **config):
        n = clients["n"]
        f = clients["f"]
        client_data = [(x, clients["client"], clients["client_args"]) for x in clients["client_ct"]] + [
            (x, clients["f_type"], clients["f_args"]) for x in clients["f_ct"]
        ]
        # Use this line for the wrongly assumed world size. It does not account for the byzaintine clients
        # num_clients = len(client_data)
        # Use this line for the correct world size
        num_clients = len(client_data) - f

        # for x in client_data:
        #     print(x)
        # exit()

        self.train_set = afl_dataset2(self.dataset_name, data_type="train")
        self.test_set = afl_dataset2(self.dataset_name, data_type="test")
        logging.info('Creating server')

        backdoor_data = [z for _x, _y, z in client_data if 'backdoor_args' in z]
        if len(backdoor_data) > 0:
            backdoor_args = backdoor_data[0]['backdoor_args']
            self.entities["server"] = config["server"](n, f, self.test_set, self.model_name, backdoor_args=backdoor_args, project_name=self.project, aux_meta_data=self.aux_meta_data, **config["server_args"])
        else:
            self.entities["server"] = config["server"](n, f, self.test_set, self.model_name, project_name=self.project, aux_meta_data=self.aux_meta_data, **config["server_args"])

        def create_client(self, pid, c_ct, client_class, client_args) -> int:
            node_id = f"c_{pid}"
            # print(client_class)
            if 'backdoor_args' in client_args:
                client_args = {**client_args, **client_args['backdoor_args']}
                del client_args['backdoor_args']
                self.entities[node_id] = client_class(pid, num_clients, self.train_set, self.model_name, **client_args)
            else:
                self.entities[node_id] = client_class(pid, num_clients, self.train_set, self.model_name, **client_args)
            self.entities[node_id].dataset_name = self.dataset_name
            self.compute_times[pid] = c_ct
            return pid

        def create_client_aux(self, pid, c_args):
            p = Thread(target=create_client, args=(self, pid, *c_args), daemon=False)
            # p.daemon = False
            p.start()
            return p
        logging.info('Creating clients')
        loading_processes = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            try:
                position = self.worker_id
                for pid, (c_ct, client_class, client_args) in enumerate(client_data):
                    
                    # print(pid, c_ct, client_class, client_args)
                    node_id = f"c_{pid}"
                    # loading_processes.append(create_client_aux(self, pid, (c_ct, client_class, client_args)))
                    loading_processes.append(executor.submit(create_client, self, pid, c_ct, client_class, client_args))

                for idx, future in tqdm(enumerate(as_completed(loading_processes)), desc='Creating clients', position=position, leave=None, total=len(loading_processes)):
                    # get the downloaded url data
                    pid = future.result()
                    logging.debug(f'{num_clients - idx} clients to be created')
                logging.info('Finished creating clients')
            except KeyboardInterrupt:
                executor.shutdown(wait=False)
                raise KeyboardInterrupt('KeyboardInterrupt 1') # Or re-raise if not in generator


        # for pid, (c_ct, client_class, client_args) in enumerate(client_data):
        #     # print(pid, c_ct, client_class, client_args)
        #     node_id = f"c_{pid}"
        #     loading_processes.append(create_client_aux(self, pid, (c_ct, client_class, client_args)))

        # [x.join() for x in loading_processes]

    def get_server(self):
        return self.entities["server"]

    def get_clients(self) -> List[Client]:
        return [item for (key, item) in self.entities.items() if key != "server"]

    def dereference(self, e_id):
        return self.entities[e_id]

    @staticmethod
    def compute_interaction_schedule(ct_data: dict, num_rounds: int) -> Tuple[List[Any], List[Any]]:
        def create_mock_client(_id, ct):
            return {
                "_id": _id,
                "ct": ct,
                "ct_left": ct,
            }

        clients = [create_mock_client(idx, c_ct) for (idx, c_ct) in ct_data.items()]
        # Make sure to sort clients! It is not ordered!
        clients = sorted(clients, key=lambda x: x['_id'])
        wall_time = 0
        events = []
        sequence = []
        for _round in range(num_rounds):
            rc = min(clients, key=lambda x: x["ct_left"])
            # Find client the finishes first
            min_ct = rc["ct_left"]
            sequence.append(rc["_id"])

            # Update the time for all the clients with the elapsed time of min_ct
            wall_time += min_ct
            events.append([rc['_id'], wall_time, min_ct, rc['ct']])
            for c in clients:
                c["ct_left"] -= min_ct

            # Perform client server interaction between server and min_ct
            # Reset compute time of min_ct
            rc["ct_left"] = rc["ct"]
            clients[rc["_id"]] = rc
        return sequence, events



    # def run_no_tasks(self, num_rounds, ct_clients=[], progress_disabled=False, position=0, add_descr=""):
    # def run_sync_tasks(self, num_rounds, ct_clients=[], progress_disabled=False, position=0, add_descr="", client_participation=1.0):


    def execute(self, num_rounds, ct_clients=[], progress_disabled=False, position=0, server_name="", client_participation=1.0, fl_type: str = 'async', batch_limit = -1, test_frequency = 25):
        interaction_sequence, interaction_events = [], []
        interaction_sequence_async, interaction_events_async = [], []
        if not ct_clients:
            ct_clients = [1] * len(self.get_clients())

        # interaction_sequence, interaction_events = self.compute_interaction_sequence(self.compute_times, num_rounds + 1)
        interaction_sequence_async, interaction_events_async = Scheduler.compute_interaction_schedule(self.compute_times, num_rounds + 1)
        # interaction_sequence = (list(self.compute_times.keys()) * (int(num_rounds / len(self.compute_times)) + 1))[
        #     :num_rounds
        # ]
        if fl_type != 'sync':
            logging.info('Running async')
            # Compute the clients server interactions

        else:
            logging.info('Running synchronous')\
            

        

        # print(self.compute_times)

        # Create entities
        clients: List[Client] = self.get_clients()
        server: Server = self.get_server()

        # Setup clients
        initial_weights = flatten(server.network)
        model_age = server.get_age()
        for c in clients:
            unflatten(c.network, initial_weights.detach().clone())

        # To keep track of the metrics
        server_metrics = []
        model_age_stats = []
        bft_telemetry = []

        # Iterate rounds

        assert fl_type in ['sync', 'async', 'semi-async']
        logging.debug(f'{server_name} is running with {fl_type=}')
        if fl_type == 'sync':
            server_metrics, bft_telemetry, interaction_events, aggregation_events  = self._sync_exec_loop(num_rounds, server, clients, client_participation, position, server_name, batch_limit = batch_limit, test_frequency=test_frequency)
        elif fl_type == 'semi-async':
            server_metrics, bft_telemetry, interaction_events, aggregation_events = self._semi_async_exec_loop(num_rounds, server, clients, client_participation, position, server_name, batch_limit = batch_limit, test_frequency=test_frequency)
            logging.warning(f'{interaction_sequence=}')
        else:
            server_metrics, model_age_stats, bft_telemetry, aggregation_events = self._async_exec_loop(num_rounds, server, clients, interaction_sequence_async, position, server_name, batch_limit=batch_limit, test_frequency=test_frequency)
            interaction_events = interaction_events_async

        return server_metrics, model_age_stats, bft_telemetry, interaction_events, aggregation_events, server.wandb_obj

    def _async_exec_loop(self, num_rounds, server:Server, clients: List[Client], interaction_sequence, position, server_name, batch_limit=-1, test_frequency=25):
        # Play all the server interactions
        logging.info('Running async loop')
        server_metrics = []
        model_age_stats = []
        last_five_loses = []
        add_descr = f"[W{position}: {wrap_name(server_name)}] "
        for update_id, client_id in enumerate(
            pbar := tqdm(interaction_sequence, position=position, leave=None, desc=add_descr)
        ):
            if update_id % test_frequency == 0:
                out = server.evaluate_model()
                last_five_loses.append(out[1])
                last_five_loses = last_five_loses[-5:]
                if np.isnan(last_five_loses).all():
                    logging.warning('Server is stopping because of too many successive NaN values during server testing')
                    break
                server_metrics.append([update_id, out[0], out[1], out[2]])
                pbar.set_description(f"{add_descr}{self.metric} = {out[0]:.2f}, Loss = {out[1]:.7f}")
                logging.info(f"[R {update_id:3d} {server_name}] {self.metric} = {out[0]:.2f}, Loss = {out[1]:.7f}")

            client: Client = clients[client_id]
            client.move_to_gpu()
            client.train(num_batches=batch_limit)

            is_byzantine = client.is_byzantine
            client_age = client.local_age
            if type(server) == Kardam:
                agg_weights, _has_aggregated = server.client_weight_dict_vec_update(client_id, client.get_model_dict_vector(), client_age, is_byzantine, client.lipschitz)
            else:
                agg_weights, _has_aggregated = server.client_weight_dict_vec_update(client_id, client.get_model_dict_vector(), client_age, is_byzantine)
            client.load_model_dict_vector(agg_weights)
            client.local_age = server.age

            # agg_weights = server.client_weight_update(client_id, client.get_weights(),client_age, is_byzantine)
            # logging.info(agg_weights.keys())
            # logging.info(agg_weights)
            # client.set_weights(agg_weights, server.get_age())
            client.move_to_cpu()
            model_age_stats.append([update_id, client.pid, client_age])

        return server_metrics, model_age_stats, server.bft_telemetry, []

    def _semi_async_exec_loop(self, num_rounds: int, server: Server, clients: List[Client], client_participation,  position = 0, server_name='', batch_limit = -1, test_frequency=5):


        # List of clients:
        # A client can be either idle (waiting) or computing.
        # clients_adm = {
        #     'idle': [],
        #     'computing': []
        # }


        has_recently_aggregated = False



        interaction_events = []
        aggregation_events = []
        wall_time = 0
        server_metrics = []
        last_five_loses = []
        schedulerCtx = SchedulerContext(clients, self.compute_times)
        server.sched_ctx = schedulerCtx # type:ignore
        # computing_clients = []
        # waiting_clients = []
        byz_clients = []
        add_descr = f"[W{position}: {wrap_name(server_name)}] "
        server_age = 0

        # Init
        agg_weight_vec: np.ndarray = server.get_model_dict_vector()
        for client in clients:
            schedulerCtx.clients_adm['computing'].append([self.compute_times[client.pid], client, False, wall_time])
            client.move_to_gpu()
            client.load_model_dict_vector(agg_weight_vec)
            client.local_age = server.get_age()
            # client.set_weights(agg_weights, server.get_age())
            client.move_to_cpu()

        num_clients = len(clients)
        k = 3 # Keep track of k number of models in total


        total_iter = num_rounds*num_clients + 1
        # agg_bound = num_clients
        agg_bound =( 2*server.f) + 1
        next_client_weights = []
        for _idx, update_id in enumerate(pbar := tqdm(range(total_iter), position=position, leave=None)):


            # Test progress
            if has_recently_aggregated and update_id % test_frequency == 0:
                has_recently_aggregated = False
                result = self.test_server(server, update_id, server_age, add_descr, server_metrics, last_five_loses, pbar, position=position, sim_time=wall_time)
                if not result:
                    # Stop training
                    break
                # out = server.evaluate_model()
                # server_metrics.append([update_id, out[0], out[1]])
                # last_five_loses.append(out[1])
                # last_five_loses = last_five_loses[-5:]
                # if np.isnan(last_five_loses).all():
                #     logging.warning('Server is stopping because of too many successive NaN values during server testing')
                #     break
                # pbar.set_description(f"{server_age} {add_descr}{self.metric} = {out[0]:.2f}, Loss = {out[1]:.7f}")

            # Next client
            # schedulerCtx.clients_adm['computing'].sort(key=lambda x: x[0])

            # client_time, next_client = schedulerCtx.clients_adm['computing'].pop(0)

            # Sort and get next client
            client_time, next_client = schedulerCtx.next_client()
            next_client : Client

            # if client_time == 0:
            #     logging.warning(f'[WARNING]\t Client {next_client.pid} has a zero time!! {self.compute_times[next_client.pid]=}')



            # Do something with the client
            next_client.move_to_gpu()
            next_client.train(num_batches=batch_limit)
            is_byzantine = next_client.is_byzantine
            c_id = next_client.get_pid()
            byz_clients.append((c_id, is_byzantine))

            # Instead of append, give it to the server?
            # @TODO: do some action after server interaction:
            # - Add client to waiting
            # - Give client weights

            # The server sends models to clients
            # The server let the scheduler know if the client should wait or not
            # We also need to know if the server aggregated or not
            
            # Log the current wall time
            logging.info(f'Wall time: {wall_time}')
            if type(server) == Kardam:
                res, has_aggregated = server.client_weight_dict_vec_update(c_id, next_client.get_model_dict_vector(),next_client.local_age, is_byzantine, next_client.lipschitz)
            else:
                res, has_aggregated = server.client_weight_dict_vec_update(c_id, next_client.get_model_dict_vector(),next_client.local_age, is_byzantine)
            # res, has_aggregated = server.client_weight_dict_vec_update(c_id, next_client.get_model_dict_vector(),next_client.local_age, is_byzantine)
            if isinstance(res, np.ndarray) or res != None:
                next_client.load_model_dict_vector(res)
                next_client.local_age = server.get_age()
                schedulerCtx.move_client_to_compute_mode(c_id) #type: ignore

            # next_client_weights.append(next_client.get_model_dict_vector())

            # next_client_weights.append(next_client.get_weights())
            next_client.move_to_cpu()



            schedulerCtx.adjust_time(client_time)
            # Update computing clients
            # for cc in schedulerCtx.clients_adm['computing']:
            #     cc[0] -= client_time
            # assert client_time <= 0.0
            wall_time += client_time
            assert schedulerCtx.wall_time == wall_time
            if has_aggregated:
                aggregation_events.append([update_id, wall_time])
                has_recently_aggregated = True
            interaction_events.append([next_client.pid, wall_time, client_time, client_time])
            # schedulerCtx.clients_adm['idle'].append(next_client)

            # # This should be a server check
            # if len(next_client_weights) == agg_bound:
            #     logging.info(f'Time to aggregate with {len(next_client_weights)} models')
            #     agg_weight_vec = server.aggregate_sync(next_client_weights, byz_clients)
            #     server_age = server.get_age()
            #     byz_clients = []
            #     next_client_weights = []
            #     for wc in schedulerCtx.clients_adm['idle']:
            #         schedulerCtx.clients_adm['computing'].append([self.compute_times[wc.pid], wc])
            #         wc.move_to_gpu()
            #         wc.load_model_dict_vector(agg_weight_vec)
            #         wc.local_age = server.get_age()
            #         # wc.set_weights(agg_weights, server.get_age())
            #         wc.move_to_cpu()
            #     waiting_clients = []


            # computing_clients.append([self.compute_times[next_client.pid], next_client])


        # logging.info(f'Overview of idle clients: {server.idle_clients}')
        return server_metrics, server.bft_telemetry, interaction_events, aggregation_events

    def _sync_exec_loop(self, num_rounds: int, server: Server, clients: List[Client], client_participation,  position = 0, server_name='', batch_limit = -1, test_frequency=5):
        test_frequency = 1
        
        server_metrics = []
        update_counter = 0 # Keep track of the number of total updates from clients
        agg_weights = server.get_model_weights()
        agg_weight_vec: np.ndarray = server.get_model_dict_vector()
        add_descr = f"[W{position}: {wrap_name(server_name)}] "
        interaction_events = []
        wall_time = 0
        num_clients = int(np.max([1, np.floor(float(len(clients)) * client_participation)]))

        num_rounds = num_rounds // num_clients
        for _idx, update_id in enumerate(pbar := tqdm(range(num_rounds + 1), position=position, leave=None, total=num_rounds*num_clients)):
            # Client selection
            selected_clients:List[Client] = np.random.choice(clients, num_clients, replace=False)  # type: ignore
            cts = [self.compute_times[x.pid] for x in selected_clients] # Get all compute times of selected clients
            round_time : float = np.max(cts) # type:ignore
            logging.info(f'Round {_idx} will take {round_time} to complete')

            # Send models to clients
            for client in clients:

                client.move_to_gpu()
                client.load_model_dict_vector(agg_weight_vec)
                client.local_age = server.get_age()
                # client.set_weights(agg_weights, server.get_age())
                client.move_to_cpu()

            # Test progress
            if update_id % test_frequency == 0:
                out = server.evaluate_model()
                server_metrics.append([update_id, out[0], out[1], out[2]])
                pbar.set_description(f"{add_descr}{self.metric} = {out[0]:.2f}, Loss = {out[1]:.7f}")

            client_weights = []
            byz_clients = []


            for _local_id, client in enumerate(selected_clients):
                client.move_to_gpu()
                client.train(num_batches=batch_limit)
                is_byzantine = client.is_byzantine
                c_id = client.get_pid()
                client_weights.append(client.get_model_dict_vector())
                # client_weights.append(client.get_weights())
                byz_clients.append((c_id, is_byzantine))
                client.move_to_cpu()
                pbar.update()

            # Aggregate
            # agg_weights = fed_avg(client_weights)

            # @TODO: Add the ability to use other algorithms than fed_avg

            agg_weight_vec = server.aggregate_sync(client_weights, byz_clients)
            # agg_weight_vec = fed_avg_vec(client_weights)
            # server.load_model_dict_vector(agg_weight_vec)
            # server.set_weights(agg_weights)
            wall_time += round_time
            interaction_events.append([0, wall_time, round_time, round_time])
            # server.incr_age()

        return server_metrics, server.bft_telemetry, interaction_events, []

    def test_server(self, server: Server , update_id, server_age,  add_descr = '', server_metrics = [], last_five_loses = [], pbar = [], backdoor=False, position: int = 0, sim_time = 0):
        # Test progress
        out = server.evaluate_model(sim_time=sim_time)
        server_metrics.append([update_id, out[0], out[1], out[2]])
        last_five_loses.append(out[1])
        last_five_loses = last_five_loses[-5:]
        if np.isnan(last_five_loses).all():
            logging.warning('Server is stopping because of too many successive NaN values during server testing')
            return False
        pbar.set_description(f"{server_age} {add_descr}{self.metric} = {out[0]:.2f}, Loss = {out[1]:.7f}")
        # @TODO: Fix using reporting functions
        # wandb.log({"update_id": update_id, "server_age": server_age, "acc": out[0], "loss": out[1], 'position': position})
        return True

    @staticmethod
    def create_scheduler(cfg: dict, worker_id: int):
        sched = Scheduler(**cfg, worker_id=worker_id)
        return sched


    @staticmethod
    def run_util(cfg_params):
        # def run_util(cfg, outfile, lock):
        """Run an experiment configuration
        @TODO: Write result to file using rlock if file is provided
        Args:
            cfg (dict): _description_

        Returns:
            List[results, config (dict)]: _description_
        """
        cfg, outfile, lock = cfg_params
        safe_cfg = dict_convert_class_to_strings(cfg)
        # lock = tqdm.get_lock()
        try:
            try:
                worker_id = int(current_process()._identity[0])
            except:
                worker_id = 0
            sched = Scheduler.create_scheduler(cfg, worker_id)
            num_rounds = cfg["num_rounds"]
            results = []

            cfg["client_participartion"] = cfg.get("client_participartion", 1.0)
            cfg["aggregation_type"] = cfg.get("aggregation_type", "async")
            cfg["eval_interval"] = cfg.get("eval_interval", 25) # Default is server eval after 25 server interactions
            cfg["client_batch_size"] = cfg.get("client_batch_size", -1) # Default is full epoch
            results = [
                [
                    # Execute returns an array: [server_metrics, model_age_stats, bft_telemetry, interaction_events]
                    *sched.execute(
                        num_rounds,
                        position=worker_id,
                        server_name=f"{safe_cfg['server']}",
                        client_participation=cfg["client_participartion"],
                        fl_type=cfg["aggregation_type"],
                        batch_limit=cfg["client_batch_size"],
                        test_frequency=cfg["eval_interval"]
                    )
                ],
                safe_cfg,
            ]
            wandb_obj = results[0][-1]
            results[0] = results[0][:-1]

            if outfile:
                if lock:
                    lock.acquire()
                completed_runs = []
                outfile: Path
                if Path(outfile).exists():
                    with open(outfile, "r") as f:
                        completed_runs = json.load(f)
                completed_runs.append(results)
                with open(outfile, "w") as f:
                    json.dump(completed_runs, f)
                if lock:
                    lock.release()
            finish_exp(wandb_obj)
            return results, wandb_obj
        except Exception as ex:
            print("Got an exception while running!!")
            print(traceback.format_exc())
            logging.error("Got an exception while running")
            logging.error(traceback.format_exc())

    @staticmethod
    def filter_executed_exps(configs: List[dict], data_file: Union[str, Path]):
        print('Running mising experiments:')
        with open(data_file, 'r') as f:
            completed_runs = json.load(f)
            keys = [x[1]['exp_id'] for x in completed_runs]
            configs = [x for x in configs if x['exp_id'] not in keys]
            # @TODO: Append to output instead of overwriting
        return configs
    
    @staticmethod
    def plot_data_distribution_by_time(configs, use_cache=False, filter_byzantine=True, cache_dir='tmp/cache', plot_dir='tmp', plot_name_prefix='data_dist'):
            from matplotlib import pyplot as plt
            import seaborn as sns

            # Make sure cache_dir and the plot_dir exists using the Path object
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            Path(plot_dir).mkdir(parents=True, exist_ok=True)

            # Iterate over all configs and generate data distribution
            for cfg_id, config in enumerate(configs):   
                print(f'Processing config {cfg_id}')  

                limit = 10
                if 'limit' in config['clients']['client_args']['sampler_args']:
                    limit = config['clients']['client_args']['sampler_args']['limit']  
                # Number of byzantines is
                num_byz = len(config['clients']['f_ct'])
                benign_clients = len(config['clients']['client_ct'])
                total_clients = benign_clients + num_byz

                cache_file = f'{cache_dir}/cache_{cfg_id}.csv'
                if not use_cache:
                    distribution_df = Scheduler.get_data_distribution(config)
                    distribution_df.to_csv(cache_file)

                distribution_df = pd.read_csv(cache_file)
                compute_times = sorted(distribution_df['compute_time'].unique())

                num_slow_clients = total_clients - (2*num_byz+ 1)
                fast_ct = compute_times[:2*num_byz+ 1]
                slow_ct = compute_times[2*num_byz+ 1:]

                distribution_df = distribution_df[distribution_df['lcount'] > 0]
                new_data = []
                for idx, row in tqdm(distribution_df.iterrows()):
                    for i in range(row['lcount']):
                        new_data.append([row['client'], row['compute_time'], row['byzantine'], row['label'], 1])

                new_df = pd.DataFrame(new_data, columns=['client', 'compute_time', 'byzantine', 'label', 'lcount'])

                # Add client type to new_df based on fast_ct and slow_ct
                new_df['client_type'] = 'fast'
                new_df.loc[new_df['compute_time'].isin(slow_ct), 'client_type'] = 'slow'
                new_df.loc[new_df['byzantine'] == 1, 'client_type'] = 'byzantine'

                # Filter out all byzanitne clients from new_df
                if filter_byzantine:
                    new_df = new_df[new_df['byzantine'] == 0]

                file_name = f'{plot_dir}/{plot_name_prefix}_cfg{cfg_id}_l{limit}.png'
                plt.figure()

                sns.displot(new_df, x='label', hue='client_type', multiple='stack')
                plt.savefig(file_name)
                plt.savefig(file_name.replace('.png', '.pdf'), bbox_inches='tight')
                plt.close()
                print(f'Graph written to {file_name}')

    @staticmethod
    def get_data_distribution(config, num_labels = 10) -> pd.DataFrame:
        worker_id = 0
        reporting_val = False
        if 'reporting' in config['server_args']:
            reporting_val = config['server_args']['reporting']
            config['server_args']['reporting'] = False
        
        sched = Scheduler.create_scheduler(config, worker_id)
        summed = 0
        import torch.nn.functional as F
        label_names = [str(x) for x in range(num_labels)]



        data = []
        cum_size = 0
        for e_id, ent in tqdm([(x,y) for x,y in sched.entities.items() if x.startswith('c')], desc='Iterating clients'):
            ent: Client
            labels = []
            compute_time = sched.compute_times[int(e_id[2:])]

            # print(f'Client {e_id} has {sched.entities}')
            # print(f'{sched.compute_times=}')

            # Print the size of the client dataset 
            # Make sure to also print the cummulative size of the dataset
            # print(f'{len(ent.train_set.dataset)=}')

            cum_size += len(ent.train_set.dataset) # type: ignore

            # Print the size of the dataset, the cummulative size and the e_id
            # print(f'{len(ent.train_set.dataset)=}, {cum_size=}, {e_id=}')

            for batch_idx, (inputs, labels_tmp) in enumerate(ent.train_set):
                labels.append(labels_tmp)
            labels = torch.cat(labels, 0).bincount()
            # padding =  torch.zeros(11)
            # labels = result = F.pad(input=labels, pad=(1), mode='constant', value=0)
            pad_size = num_labels - len(labels)
            m = torch.nn.ConstantPad1d((0,pad_size), 0)
            labels = m(labels)

            for l_count, l_name in zip(labels.tolist(), label_names):
                data.append([e_id, compute_time, ent.is_byzantine, int(l_name), int(l_count)])


            # data.append([e_id, ent.is_byzantine, *labels.tolist()])

            


            # labels = [x[1][1] for x in enumerate(ent.train_set)]
            summed += len(ent.train_set.dataset.indices)
        cnames = ['client', 'compute_time', 'byzantine', 'label', 'lcount']
        df = pd.DataFrame(data, columns= cnames)

        # Restore the reporting value
        if 'reporting' in config['server_args']:
            config['server_args']['reporting'] = reporting_val
        return df


    @staticmethod
    def run_multiple(list_of_configs, pool_size=5, outfile: Union[str, Path] = 'data.json', clear_file=False, multi_thread = True, autocomplete = False):
        logging.basicConfig(format='%(asctime)s.%(msecs)-3d - %(levelname)-8s - %(processName)-2s :  %(message)s', level=logging.DEBUG, filename='debug.log', datefmt="%H:%M:%S")
        logging.info(f'<== Starting execution of {len(list_of_configs)} experiments at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} ==>')
        if autocomplete:
            list_of_configs = Scheduler.filter_executed_exps(list_of_configs, outfile)
        pool_size = min(pool_size, len(list_of_configs))
        # install_mp_handler()
        
        if clear_file and outfile and Path(outfile).exists():
            logging.info(f'Clearing existing output file "{Path(outfile).name}"')
            Path(outfile).unlink()
        outfile = Path(outfile)
        start_time = time.time()
        outputs = []
        lock = tqdm.get_lock()
        lock = None
        cfg_args = [(x, outfile, lock) for x in list_of_configs]
        logging.info(f"Pool size = {pool_size}")
        logging.info(f'Run multi-threaded ? {multi_thread}')
        wandb.setup()
        def init_func(args):
            multiprocessing.current_process().name = multiprocessing.current_process().name.replace('ForkPoolWorker-', 'W')
            tqdm.set_lock(args)
        if pool_size > 1 and multi_thread:
            with Pool(pool_size, initializer=init_func, initargs=(tqdm.get_lock(),)) as pool:

                # @TODO: Make sure the memory is dealocated when the task is finished. Currently is accumulating memory with lots of tasks?
                outputs = [
                    x
                    for x in tqdm(
                        pool.imap_unordered(Scheduler.run_util, cfg_args),
                        total=len(list_of_configs),
                        position=0,
                        leave=None,
                        desc="Total",
                    )
                    if x
                ]
        else:
            print('Running single-threaded')
            outputs = [Scheduler.run_util(x) for x in cfg_args]
        # print(f'{outputs=}')
        if outputs[-1]:

            last_wandb = outputs[-1][-1]
            last_wandb = wandb.init(reinit=True)
        print(f'{last_wandb=}')
        print(f"--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---")
        
        finish_reporting(last_wandb, outfile.stem, f'--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---')
        return outputs
