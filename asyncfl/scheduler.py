#
# from .server import Server
import copy
import json
from multiprocessing import Lock, Manager, Pool, current_process, RLock, Process
from multiprocessing.pool import AsyncResult
from pathlib import Path
import traceback
from typing import List, Union
import torch
from tqdm.auto import tqdm
from os import getpid
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
from asyncfl.dataloader import afl_dataset2
from .network import get_model_by_name, model_gradients, flatten, unflatten_b, unflatten_g, flatten_g, unflatten
from asyncfl.network import flatten
from .server import Server
from .client import Client
from .task import Task
import numpy as np
import time
import gc
import asyncio
import random
from threading import Thread

def dict_convert_class_to_strings(dictionary: dict):
    d = copy.deepcopy(dictionary)
    d['clients']['client'] = d['clients']['client'].__name__
    d['clients']['f_type'] = d['clients']['f_type'].__name__
    d['server'] = d['server'].__name__
    return d

class PoolManager():

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None) -> None:
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
                self.active_tasks.append(
                    (self.pool.apply_async(func, args), cap))

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
        if 'learning_rate' not in config['server_args']:
            config['server_args']['learning_rate'] = 0.005
        self.create_entities(**config)
        self.dataset = None

    def create_entities(self, clients, **config):
        n = clients['n']
        f = clients['f']
        client_data = [(x, clients['client'], clients['client_args']) for x in clients['client_ct']
                       ] + [(x, clients['f_type'], clients['f_args']) for x in clients['f_ct']]
        num_clients = len(client_data)

        self.train_set = afl_dataset2(self.dataset_name,  data_type="train")
        self.test_set = afl_dataset2(self.dataset_name,  data_type="test")
        self.entities['server'] = config['server'](n, f,
            self.test_set, self.model_name, **config['server_args'])

        def create_client(self, pid, c_ct, client_class, client_args):
            node_id = f'c_{pid}'
            self.entities[node_id] = client_class(
                pid, num_clients, self.train_set, self.model_name, **client_args)
            self.compute_times[pid] = c_ct

        def create_client_aux(self, pid, c_args):
            p = Thread(target=create_client, args=(
                self, pid, *c_args), daemon=False)
            # p.daemon = False
            p.start()
            return p
        loading_processes = []
        for pid, (c_ct, client_class, client_args) in enumerate(client_data):
            # print(pid, c_ct, client_class, client_args)
            node_id = f'c_{pid}'
            loading_processes.append(create_client_aux(
                self, pid, (c_ct, client_class, client_args)))

        [x.join() for x in loading_processes]

    def get_server(self):
        return self.entities['server']

    def get_clients(self):
        return [item for (key, item) in self.entities.items() if key != 'server']

    def dereference(self, e_id):
        return self.entities[e_id]

    def compute_interaction_sequence(self, ct_data, num_rounds):
        def create_mock_client(_id, ct):
            return {
                '_id': _id,
                'ct': ct,
                'ct_left': ct,
            }

        clients = [create_mock_client(idx, c_ct)
                   for (idx, c_ct) in ct_data.items()]
        sequence = []
        for _round in range(num_rounds):
            rc = min(clients, key=lambda x: x['ct_left'])
            # Find client the finishes first
            min_ct = rc['ct_left']
            sequence.append(rc['_id'])

            # Update the time for all the clients with the elapsed time of min_ct
            for c in clients:
                c['ct_left'] -= min_ct + 1


            # Perform client server interaction between server and min_ct
            # Reset compute time of min_ct
            rc['ct_left'] = rc['ct']
            clients[rc['_id']] = rc
        return sequence

    def run_sync_tasks(self, num_rounds, ct_clients=[], progress_disabled=False, position=0, add_descr='', client_participation=1.0):
        clients: List[Client] = self.get_clients()
        server: Server = self.get_server()

        def train_client(self, client: Client, local_id, num_batches=-1):
            client.move_to_gpu()
            client.train(num_batches=num_batches)
            c_gradients, c_buffers, age = client.get_gradient_vectors()
            self.gradient_responses[local_id] = c_gradients
            self.buffer_responses[local_id] = c_buffers
            client.move_to_cpu()

        initial_weights = flatten(server.network)
        model_age = server.get_age()
        for c in clients:
            unflatten(c.network, initial_weights.detach().clone())

        server_metrics = []
        update_counter = 0
        for idx_, update_id in enumerate(pbar := tqdm(range(num_rounds+1), position=position, leave=None)):
            # print(f'Round {update_id}')
            num_clients = int(
                np.max([1, np.floor(float(len(clients))*client_participation)]))
            selected_clients = np.random.choice(
                clients, num_clients, replace=False)
            # print(f'Client participation: {num_clients}')
            if update_id % 5 == 0:
                out = server.evaluate_accuracy()
                server_metrics.append([update_id, out[0], out[1]])
                pbar.set_description(
                    f'{add_descr}Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}')
            gradients = []
            buffers = []

            update_counter += len(clients)
            training_processes = []
            for local_id, client in enumerate(selected_clients):

                client.move_to_gpu()
                client.train(num_batches=1)
                c_gradients, c_buffers, lipschitz, age = client.get_gradient_vectors()
                gradients.append(c_gradients)
                buffers.append(c_buffers)
                client.move_to_cpu()
            # [x.join() for x in training_processes]

            # agv_gradient = np.mean(self.gradient_responses, axis=0)
            agv_gradient = np.mean(gradients, axis=0)
            # if not any(self.buffer_responses):
            #     avg_buffers = []
            # else:
            if buffers[0] == []:
                avg_buffers = []
            else:
                stacked = torch.stack(buffers)
            #     # stacked = torch.stack(self.buffer_responses)
                avg_buffers = torch.mean(stacked, dim=0)
            unflatten_b(server.network, avg_buffers)
            new_model_weights_vector = server.client_update(
                client.get_pid(), agv_gradient, lipschitz, server.get_age())
            # new_model_weights_vector = server.get_model_weights()
            for client in clients:
                client.move_to_gpu()
                client.set_weight_vectors(
                    new_model_weights_vector.cpu().numpy(), server.get_age())
                client.move_to_cpu()

                # unflatten_b(server.network, c_buffers)
                # new_model_weights_vector = server.client_update(client.get_pid(), c_gradients, age)
                # client.set_weight_vectors(new_model_weights_vector.cpu().numpy(), server.get_age())
                # client.move_to_cpu()
        return server_metrics, []

    def run_no_tasks(self, num_rounds, ct_clients=[], progress_disabled=False, position=0, add_descr=''):
        """
        @TODO: Put dict of interaction sequence as argument
        @TODO: Save participation statistics of the clients and plot this in a graph.
        @TODO: Keep track of the model staleness (age), and plot this in a graph
        @TODO: Create non-IID scenario
        @TODO: Make sure that the server loads the full test set and the clients the splitted train set
        @TODO: Show that non-IID and skewed compute time are bad for the accuracy
        @TODO: Break BASGD with a frequency attack
        """
        if not ct_clients:
            ct_clients = [1] * len(self.get_clients())

        interaction_sequence = self.compute_interaction_sequence(
            self.compute_times, num_rounds+1)

        # c_ids = list(self.compute_times.keys())
        # random.shuffle(c_ids)
        # interaction_sequence = (list(self.compute_times.keys())*(int(num_rounds / len(self.compute_times))+1))[:num_rounds]


        # seqs = [self.compute_interaction_sequence(
        #     self.compute_times, num_rounds+1) for x in range(10)]
        clients: List[Client] = self.get_clients()
        server: Server = self.get_server()

        initial_weights = flatten(server.network)
        model_age = server.get_age()
        for c in clients:
            unflatten(c.network, initial_weights.detach().clone())

        # To keep track of the metrics
        server_metrics = []
        model_age_stats = []

        use_weight_avg = False

        # Play all the server interactions
        for update_id, client_id in enumerate(pbar := tqdm(interaction_sequence, position=position, leave=None, desc=add_descr)):

            if update_id % 25 == 0:
                out = server.evaluate_accuracy()
                server_metrics.append([update_id, out[0], out[1]])
                pbar.set_description(
                    f'{add_descr}Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}')
            client: Client = clients[client_id]

            client.move_to_gpu()
            client.train(num_batches=1)

            if use_weight_avg:
                server.set_weights(client.network.state_dict())
            else:
                c_gradients, c_buffers, lipschitz, age = client.get_gradient_vectors()
                server.incr_age()
                gradient_age = server.get_age() - age
                model_age_stats.append([update_id, client.pid, gradient_age])
                unflatten_b(server.network, c_buffers)
                new_model_weights_vector = server.client_update(
                    client.get_pid(), c_gradients, lipschitz, gradient_age)
                client.set_weight_vectors(
                    new_model_weights_vector.cpu().numpy(), server.get_age())
            client.move_to_cpu()

        return server_metrics, model_age_stats

        # Plot data

    def run_with_tasks_old(self):
        """
        Deprecated function
        """
        raise DeprecationWarning(f"Function '{__name__}' is deprecated")
        # Create task list
        tasks = []
        task: Task = Task(self.get_clients()[0], Client.get_pid)
        task()
        t2 = Task(self.get_clients()[-1],
                  Client.print_pid_and_var, "hello world")
        t2()
        t3 = Task(self.get_clients()[0], Client.train,
                  self.get_clients()[0].get_weights())
        t_c1_join = Task(self.get_server(), Server.client_join,
                         self.get_clients()[0])
        t_c1_train = Task(self.get_clients()[0], Client.train)
        t_c1_update = Task(self.get_server(),
                           Server.client_update, self.get_clients()[0])
        tasks = [t_c1_join] + [t_c1_train, t_c1_update]*1000
        for t in tasks:
            t()
        for c in self.server.clients:
            self.server.client_join(c)
        # Run 5 times
        for i in range(5):
            print(f'Running iter {i}')
        return
        # current_model = self.server.client_join(self.clients[0])
        # self.clients
        # Get initial model weights
        current_weights = self.server.get_model_weights()
        # while True:
        grad = self.clients[0].train(current_weights)
        print(grad)
        # for c in self.clients:
        #     c.train()

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
            num_rounds = cfg['num_rounds']
            results = []
            if 'aggregation_type' in cfg and cfg['aggregation_type'] == 'sync':
                # Run synchronous scheduler
                worker_id = int(current_process()._identity[0])

                # worker_id = 1
                cfg['client_participartion'] = cfg.get(
                    'client_participartion', 1.0)
                # print('Running SYNC scheduler')
                results = [[*sched.run_sync_tasks(num_rounds, position=worker_id, add_descr=f'[Worker {worker_id}] ', client_participation=cfg['client_participartion'])], safe_cfg]
            else:
                # Default: Run asyn scheduler
                # print('Running ASYNC scheduler')
                worker_id = int(current_process()._identity[0])
                results = [[*sched.run_no_tasks(num_rounds, position=worker_id, add_descr=f'[Worker {worker_id}] ')], safe_cfg]
            if outfile:
                if lock:
                    lock.acquire()
                completed_runs = []
                outfile: Path
                if Path(outfile).exists():
                    with open(outfile, 'r') as f:
                        completed_runs = json.load(f)
                completed_runs.append(results)
                with open(outfile, 'w') as f:
                    json.dump(completed_runs, f)
                if lock:
                    lock.release()
                    
            return results
        except Exception as ex:
            print('Got an exception while running!!')
            print(traceback.format_exc())

    @staticmethod
    def run_multiple(list_of_configs, pool_size=5, outfile: Union[str, Path, None] = None, clear_file = False):
        if clear_file and outfile and Path(outfile).exists():
            Path(outfile).unlink()
        start_time = time.time()
        outputs = []
        # m = Manager()
        lock = tqdm.get_lock()
        lock = None
        print(f'Pool size = {pool_size}')
        pool = Pool(pool_size, initializer=tqdm.set_lock,
                    initargs=(tqdm.get_lock(),))
        cfg_args = [(x, outfile, lock) for x in list_of_configs]
        # @TODO: Make sure the memory is dealocated when the task is finished. Currently is accumulating memory with lots of tasks
        outputs = [x for x in tqdm(pool.imap_unordered(Scheduler.run_util, cfg_args), total=len(
            list_of_configs), position=0, leave=None, desc='Total') if x]
        print(
            f"--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---")
        return outputs

    @staticmethod
    def run_pm(list_of_configs, pool_size=5):
        start_time = time.time()
        pm = PoolManager(pool_size, initializer=tqdm.set_lock,
                         initargs=(tqdm.get_lock(),))
        pbar = tqdm(total=len(list_of_configs),
                    position=0, leave=None, desc='Total')
        for cfg in list_of_configs:
            cfg['task_cap'] = cfg.get('task_cap', 1.0/pool_size)
            pm.add_task(Scheduler.run_util, [cfg], cfg['task_cap'])

        pm.run(pbar=pbar)
        results = [x.get() for x in pm.get_results()]
        print(
            f"--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---")
        return results

    @staticmethod
    def run_util_sync(cfg):
        """
        Deprecated function
        """
        raise DeprecationWarning(f"Function '{__name__}' is deprecated")
        sched = Scheduler(**cfg)
        num_rounds = cfg['num_rounds']
        # worker_id = int(current_process()._identity[0])
        worker_id = 1
        cfg['client_participartion'] = cfg.get('client_participartion', 1.0)
        return [sched.run_sync_tasks(num_rounds, position=worker_id, add_descr=f'[Worker {worker_id}] ', client_participation=cfg['client_participartion']), cfg]

    @staticmethod
    def run_sync(list_of_configs, pool_size=5):
        """
        Deprecated function
        """
        raise DeprecationWarning(f"Function '{__name__}' is deprecated")
        return [Scheduler.run_util_sync(cfg) for cfg in tqdm(list_of_configs, total=len(list_of_configs), position=0, leave=None, desc='Total')]
        # return [Scheduler.run_util_sync(cfg) for cfg in list_of_configs]
