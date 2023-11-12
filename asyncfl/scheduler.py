from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime
import inspect
import json
from multiprocessing import Lock, Pool, current_process, RLock, Process
import multiprocessing
from multiprocessing.pool import AsyncResult
from pathlib import Path
import traceback
from typing import Any, List, Tuple, Union
import torch
from tqdm.auto import tqdm
import copy
from asyncfl.dataloader import afl_dataset2
from .network import flatten, unflatten_b, unflatten
from asyncfl.network import flatten
from .server import Server, fed_avg, fed_avg_vec
from .client import Client
from .task import Task
import numpy as np
import time
from threading import Thread
import logging
import asyncio
from .kardam import Kardam

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
    def __init__(self, dataset_name: str, model_name: str, **config):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.clients: List[Client] = []
        self.entities = {}
        self.compute_times = {}
        if "learning_rate" not in config["server_args"]:
            config["server_args"]["learning_rate"] = 0.005
        self.create_entities(**config)
        self.dataset = None

    def create_entities(self, clients, **config):
        n = clients["n"]
        f = clients["f"]
        client_data = [(x, clients["client"], clients["client_args"]) for x in clients["client_ct"]] + [
            (x, clients["f_type"], clients["f_args"]) for x in clients["f_ct"]
        ]
        num_clients = len(client_data)

        self.train_set = afl_dataset2(self.dataset_name, data_type="train")
        self.test_set = afl_dataset2(self.dataset_name, data_type="test")
        logging.info('Creating server')
        self.entities["server"] = config["server"](n, f, self.test_set, self.model_name, **config["server_args"])

        def create_client(self, pid, c_ct, client_class, client_args) -> int:
            node_id = f"c_{pid}"
            # print(client_class)
            self.entities[node_id] = client_class(pid, num_clients, self.train_set, self.model_name, **client_args)
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
                for pid, (c_ct, client_class, client_args) in enumerate(client_data):
                    # print(pid, c_ct, client_class, client_args)
                    node_id = f"c_{pid}"
                    # loading_processes.append(create_client_aux(self, pid, (c_ct, client_class, client_args)))
                    loading_processes.append(executor.submit(create_client, self, pid, c_ct, client_class, client_args))

                for idx, future in enumerate(as_completed(loading_processes)):
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
        if fl_type != 'sync':
            logging.info('Running async')
            # Compute the clients server interactions
            if not ct_clients:
                ct_clients = [1] * len(self.get_clients())

            # interaction_sequence, interaction_events = self.compute_interaction_sequence(self.compute_times, num_rounds + 1)
            interaction_sequence, interaction_events = Scheduler.compute_interaction_schedule(self.compute_times, num_rounds + 1)
            # interaction_sequence = (list(self.compute_times.keys()) * (int(num_rounds / len(self.compute_times)) + 1))[
            #     :num_rounds
            # ]
        else:
            logging.info('Running synchronous')
        
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
        if fl_type == 'sync':
            server_metrics, bft_telemetry, interaction_events = self._sync_exec_loop(num_rounds, server, clients, client_participation, position, server_name, batch_limit = batch_limit, test_frequency=test_frequency)
        elif fl_type == 'semi-async':
            server_metrics, bft_telemetry, interaction_events = self._semi_async_exec_loop(num_rounds, server, clients, client_participation, position, server_name, batch_limit = batch_limit, test_frequency=test_frequency)
        else:
            server_metrics, model_age_stats, bft_telemetry = self._async_exec_loop(num_rounds, server, clients, interaction_sequence, position, server_name, batch_limit=batch_limit, test_frequency=test_frequency)


        return server_metrics, model_age_stats, bft_telemetry, interaction_events

    def _async_exec_loop(self, num_rounds, server:Server, clients: List[Client], interaction_sequence, position, server_name, batch_limit=-1, test_frequency=25):
        # Play all the server interactions
        logging.info('Running async loop')
        server_metrics = []
        model_age_stats = []
        last_five_loses = []
        add_descr = f"[W{position}: {server_name}] "
        for update_id, client_id in enumerate(
            pbar := tqdm(interaction_sequence, position=position, leave=None, desc=add_descr)
        ):
            if update_id % test_frequency == 0:
                out = server.evaluate_accuracy()
                last_five_loses.append(out[1])
                last_five_loses = last_five_loses[-5:]
                if np.isnan(last_five_loses).all():
                    logging.warning('Server is stopping because of too many successive NaN values during server testing')
                    break
                server_metrics.append([update_id, out[0], out[1]])
                pbar.set_description(f"{add_descr}Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}")
                logging.info(f"[R {update_id:3d} {server_name}] Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}")
            
            client: Client = clients[client_id]
            client.move_to_gpu()
            client.train(num_batches=batch_limit)
            
            is_byzantine = client.is_byzantine
            client_age = client.local_age
            if type(server) == Kardam:
                agg_weights : np.ndarray = server.client_weight_dict_vec_update(client_id, client.get_model_dict_vector(), client_age, is_byzantine, client.lipschitz)
            else:
                agg_weights : np.ndarray = server.client_weight_dict_vec_update(client_id, client.get_model_dict_vector(), client_age, is_byzantine)
            client.load_model_dict_vector(agg_weights)
            client.local_age = server.age

            # agg_weights = server.client_weight_update(client_id, client.get_weights(),client_age, is_byzantine)
            # logging.info(agg_weights.keys())
            # logging.info(agg_weights)
            # client.set_weights(agg_weights, server.get_age())
            client.move_to_cpu()
            model_age_stats.append([update_id, client.pid, client_age])

        return server_metrics, model_age_stats, server.bft_telemetry

    def _semi_async_exec_loop(self, num_rounds: int, server: Server, clients: List[Client], client_participation,  position = 0, server_name='', batch_limit = -1, test_frequency=5):
        

        # List of clients:
        # A client can be either idle (waiting) or computing.
        # clients_adm = {
        #     'idle': [],
        #     'computing': []
        # }


        class SchedulerContext():
            def __init__(self, clients : List[Client], compute_times: dict) -> None:
                self.clients_adm = {
                    'idle': [],
                    'computing': []
                }
                self.compute_times = compute_times
                self.clients = clients
            
            def send_model_to_client(self, client_id: int, model_vec: np.ndarray, model_age: int):
                c = next((x for x in self.clients if x.pid == client_id), None)
                assert c is not None
                c.load_model_dict_vector(model_vec)
                # logging.info(f'[CTX] sending model to client {c.pid}  ({client_id}) with model age {model_age}')
                c.local_age = model_age
            
            def move_client_to_idle_mode(self, client_id: int):
                c = next((x for x in self.clients if x.pid == client_id), None)
                assert c is not None
                self.clients_adm['idle'].append(c)

            def move_client_to_compute_mode(self, client_id: int):
                c = next((x for x in self.clients if x.pid == client_id), None)
                assert c is not None
                # logging.info(f"Filtering pid {c.pid}, {client_id} from {[x.pid for x in self.clients_adm['idle']]}")
                self.clients_adm['idle'] = [x for x in self.clients_adm['idle'] if x.pid != c.pid]
                # logging.info(f'Moving client {c.pid} ({client_id}) to compute mode with ct of {self.compute_times[c.pid]}')
                self.clients_adm['computing'].append([self.compute_times[c.pid], c])
            



        interaction_events = []
        wall_time = 0
        server_metrics = []
        schedulerCtx = SchedulerContext(clients, self.compute_times)
        server.sched_ctx = schedulerCtx # type:ignore
        # computing_clients = []
        # waiting_clients = []
        byz_clients = []
        add_descr = f"[W{position}: {server_name}] "
        server_age = 0

        # Init
        agg_weight_vec: np.ndarray = server.get_model_dict_vector()
        for client in clients:
            schedulerCtx.clients_adm['computing'].append([self.compute_times[client.pid], client])
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
            if update_id % test_frequency == 0:
                out = server.evaluate_accuracy()
                server_metrics.append([update_id, out[0], out[1]])
                pbar.set_description(f"{server_age} {add_descr}Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}")

            # Next client
            schedulerCtx.clients_adm['computing'].sort(key=lambda x: x[0])

            # logging.info(f'{computing_clients=}')
            # logging.info(f"{len(schedulerCtx.clients_adm['computing'])=} && {len(schedulerCtx.clients_adm['idle'])=}")
            client_time, next_client = schedulerCtx.clients_adm['computing'].pop(0)
            # logging.info(f'{next_client.local_age=} {client_time=}, {next_client.pid=}')



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
            server.client_weight_dict_vec_update(c_id, next_client.get_model_dict_vector(),next_client.local_age, is_byzantine)
            # next_client_weights.append(next_client.get_model_dict_vector())

            # next_client_weights.append(next_client.get_weights())
            next_client.move_to_cpu()




            # Update computing clients
            for cc in schedulerCtx.clients_adm['computing']:
                cc[0] -= client_time
            assert client_time <= 0.0
            wall_time += client_time
            # logging.info(f'Make next time step of {client_time} units')
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



        return server_metrics, server.bft_telemetry, interaction_events

    def _sync_exec_loop(self, num_rounds: int, server: Server, clients: List[Client], client_participation,  position = 0, server_name='', batch_limit = -1, test_frequency=5):
        server_metrics = []
        update_counter = 0 # Keep track of the number of total updates from clients
        agg_weights = server.get_model_weights()
        agg_weight_vec: np.ndarray = server.get_model_dict_vector()
        add_descr = f"[W{position}: {server_name}] "
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
                out = server.evaluate_accuracy()
                server_metrics.append([update_id, out[0], out[1]])
                pbar.set_description(f"{add_descr}Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}")
            
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
        
        return server_metrics, server.bft_telemetry, interaction_events




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
            sched = Scheduler(**cfg)
            num_rounds = cfg["num_rounds"]
            results = []
            try:
                worker_id = int(current_process()._identity[0])
            except:
                worker_id = 0
            cfg["client_participartion"] = cfg.get("client_participartion", 1.0)
            cfg["aggregation_type"] = cfg.get("aggregation_type", "async")
            cfg["eval_interval"] = cfg.get("eval_interval", 25) # Default is server eval after 25 server interactions
            cfg["client_batch_size"] = cfg.get("client_batch_size", -1) # Default is full epoch
            results = [
                [
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

            return results
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
        start_time = time.time()
        outputs = []
        lock = tqdm.get_lock()
        lock = None
        cfg_args = [(x, outfile, lock) for x in list_of_configs]
        logging.info(f"Pool size = {pool_size}")
        logging.info(f'Run multi-threaded ? {multi_thread}')

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
        print(f"--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---")
        logging.info(f"--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---")

        return outputs
