#
# from .server import Server
from multiprocessing import Pool, current_process, RLock, Process
from typing import List
import torch
from tqdm.auto import tqdm
from os import getpid
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .network import get_model_by_name, model_gradients, flatten, unflatten_b, unflatten_g, flatten_g, unflatten
from asyncfl.network import flatten
from .server import Server
from .client import Client
from .task import Task
import asyncio
import random
from threading import Thread
class Scheduler:

    def __init__(self, dataset_name: str, model_name: str, **config):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.clients: List[Client] = []
        self.entities = {}
        self.compute_times = {}
        self.entities['server'] = config['server'](dataset_name, model_name, **config['server_args'])
        self.create_entities(**config)

    # def __init__(self, server_class, client_class, num_clients, dataset_name: str, config = None, server_args = {}, client_args = {}):
    #
    #
    #     # self.server = server_class(dataset_name)
    #     self.dataset_name = dataset_name
    #
    #
    #
    #     self.clients: List[Client] = []
    #     # self.create_entities(client_class, num_clients, config)
    #
    #     self.entities = {}
    #     self.entities['server'] = config['server'](dataset_name, **config['server_args'])
    #
    #     # Create benign clients
    #     self.create_entities(client_class, num_clients, config, client_args)
    #
    #     # Create Byzantine clients
    #     self.create_entities(client_class, num_clients, config, client_args)

    def create_entities(self, clients, **config):
        n = clients['n']
        f = clients['f']
        # Create normal workers
        # client_data = [(x, clients['f_type'], clients['f_args']) for x in clients['f_ct']] + [(x, clients['client'], clients['client_args']) for x in clients['client_ct']]
        client_data = [(x, clients['client'], clients['client_args']) for x in clients['client_ct']] + [(x, clients['f_type'], clients['f_args']) for x in clients['f_ct']]
        num_clients = len(client_data)


        def create_client(self, pid, c_ct, client_class, client_args):
            node_id = f'c_{pid}'
            # print(f'Starting node: {node_id}')
            self.entities[node_id] = client_class(pid, num_clients, self.dataset_name, self.model_name, **client_args)
            self.compute_times[pid] = c_ct
        
        # async def create_all_clients(self, client_data):
        #     # print('All gather ')
        #     await asyncio.gather(*[create_client(self, pid, *c_args) for (pid, c_args) in enumerate(client_data)])
        

        def create_client_aux(self, pid, c_args):
            p = Thread(target=create_client, args=(self, pid, *c_args), daemon=False)
            # p.daemon = False
            p.start()
            return p        
        loading_processes = []
        for pid, (c_ct, client_class, client_args) in enumerate(client_data):
            # print(pid, c_ct, client_class, client_args)
            node_id = f'c_{pid}'
            loading_processes.append(create_client_aux(self, pid, (c_ct, client_class, client_args)))

        [x.join() for x in loading_processes]

            # self.entities[node_id] = client_class(pid, num_clients, self.dataset_name, self.model_name, **client_args)
            # self.compute_times[pid] = c_ct

        # with Pool(5) as pp:
        #     pp.ma(create_client, )
        # pool = Pool(pool_size, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        # # pbar = tqdm(total=len(list_of_configs))
        # # @TODO: Make sure the memory is dealocated when the task is finished. Currently is accumulating memory with lots of tasks
        # outputs = [x for x in tqdm(pool.imap_unordered(Scheduler.run_util, list_of_configs), total=len(list_of_configs), position=0, leave=None, desc='Total')]

        # asyncio.run(create_all_clients(self, client_data))

        # for pid, (c_ct, client_class, client_args) in enumerate(client_data):
        #     # print(pid, c_ct, client_class, client_args)
        #     node_id = f'c_{pid}'
        #     self.entities[node_id] = client_class(pid, num_clients, self.dataset_name, self.model_name, **client_args)
        #     self.compute_times[pid] = c_ct

    # def create_entities(self, client_class, n, config = None, client_args = {}):
    #     """
    #     Client class A
    #     Client class F
    #     N
    #     F
    #     Client A Args
    #     Client F Args
    #     Compute times clients A
    #     Compute times clients F
    #     """
    #     compute_times = []
    #     if (not config):
    #         compute_times = [[x, 1] for x in range(n)]
    #     print(compute_times)
    #
    #     for pid, ct in compute_times:
    #         self.entities[f'c_{pid}'] = client_class(pid, self.dataset_name, **client_args)
    #         # self.clients.append(client_class(pid, self.dataset_name))
    #     # print(self.clients)

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

        clients = [create_mock_client(idx, c_ct) for (idx, c_ct) in ct_data.items()]
        sequence = []
        for _round in range(num_rounds):
            rc = min(clients, key=lambda x: x['ct_left'])
            # Find client the finishes first
            min_ct = rc['ct_left']
            sequence.append(rc['_id'])

            # Update the time for all the clients with the elapsed time of min_ct
            for c in clients:
                c['ct_left'] -= min_ct

            # Perform client server interaction between server and min_ct
            # Reset compute time of min_ct
            rc['ct_left'] = rc['ct']
            clients[rc['_id']] = rc
        return sequence

    def run_no_tasks(self, num_rounds, ct_clients = [], progress_disabled = False, position=0, add_descr=''):
        """
        @TODO: Put dict of interaction sequence as argument
        @TODO: Save accuracy and loss and plot this in a graph
        @TODO: Save participation statistics of the clients and plot this in a graph.
        @TODO: Keep track of the model staleness (age), and plot this in a graph
        @TODO: Clean up the code
        @TODO: Implement BASGD (Mainly copying code)
        @TODO: Create non-IID scenario
        @TODO: Make sure that the server loads the full test set and the clients the splitted train set
        @TODO: Show that non-IID and skewed compute time are bad for the accuracy
        @TODO: Break BASGD with a frequency attack
        """
        # print('Running without any tasks wrapper')
        # print(f'Using position {position}')
        if not ct_clients:
            ct_clients = [1] * len(self.get_clients())

        interaction_sequence = self.compute_interaction_sequence(self.compute_times, num_rounds+1)
        # print('Starting with every client joining to the server --> Get model')

        clients: List[Client] = self.get_clients()
        server: Server = self.get_server()

        # initial_weights = server.get_model_weights()
        initial_weights = flatten(server.network)
        model_age = server.get_age()
        for c in clients:
            # c.set_weights(initial_weights, model_age)
            unflatten(c.network, initial_weights.detach().clone())
            # c.set_weights(initial_weights, model_age)

        # To keep track of the metrics
        server_metrics = []

        # Play all the server interactions
        # with tqdm() as bar:
        for update_id, client_id in enumerate(pbar := tqdm(interaction_sequence, position=position, leave=None, desc=add_descr)):
            if update_id % 50 == 0:
                out = server.evaluate_accuracy()
                server_metrics.append([update_id, out[0], out[1]])
                pbar.set_description(f'{add_descr}Accuracy = {out[0]:.2f}%, Loss = {out[1]:.7f}')
            client: Client = clients[client_id]
            client.train(num_batches=1)


            # client_grad_data = client.get_gradients()
            # new_model_weights = server.client_update(client.get_pid(), *client_grad_data)
            # client.set_weights(new_model_weights, server.get_age())


            # This works realy well!!
            # c_model_weights = client.get_weights()
            # server.set_weights(c_model_weights)
            # new_model_weights = server.get_model_weights()
            # client.set_weights(new_model_weights, server.get_age())


            # New attempt by copying the buffers as well


            # c_weights, c_buffers = client.get_weight_vectors()
            # unflatten_b(server.network, c_buffers)

            # unflatten(server.network, c_weights)


            c_gradients, c_buffers, age = client.get_gradient_vectors()
            unflatten_b(server.network, c_buffers)
            new_model_weights_vector = server.client_update(client.get_pid(), c_gradients, age)
            client.set_weight_vectors(new_model_weights_vector.cpu().numpy(), server.get_age())


            # Attempt to only use paramters and not state_dict
            # c_param = client.network.parameters()

            # for target_param, param in zip(server.network.parameters(), c_param):
            #     target_param.data.copy_(param.data)

            # for target_param, param in zip(client.network.parameters(), server.network.parameters()):
            #     target_param.data.copy_(param.data)

            # new_model_weights = server.get_model_weights()
            # client.set_weights(new_model_weights, server.get_age())


            # c_weights = flatten(client.network).detach().clone()
            # s_weights = flatten(server.network).detach().clone()
            # s_grad_flat = torch.zeros_like(c_weights)
            # flatten_g(client.network, s_grad_flat)
            # s_grad_flat = s_grad_flat.detach().clone()
            # diff1 = c_weights - s_weights
            # diff2 = s_weights - c_weights


            # c_weights = client.get_weights()
            # c_grad = torch.from_numpy(client.get_gradients()[0])
            # cw_flat = flatten(client.network).detach()
            # s_flat = flatten(server.network).detach()

            # diff_flat = cw_flat - s_flat
            # diff_flat2 = s_flat - cw_flat


            # server.set_weights(c_weights)
            # new_model_weights = server.get_model_weights()
            # s2_flat = flatten(server.network)


            # # grad_data = client.get_gradients()
            # # c_weights = flatten(client.network)
            # # s_weights = flatten(server.network)
            # # t_grad_data = torch.from_numpy(grad_data[0])
            # # dff = c_weights - s_weights
            # # # c_grad = 
            # # server.set_weights(c_weights)
            # new_model_weights = server.client_update(client.get_pid(), *grad_data)
            # # new_model_weights = server.get_model_weights()
            # client.set_weights(new_model_weights, server.get_age())
            # server.client_update(client)
        # pbar.close()
        # print(server_metrics)

        return server_metrics

        # Plot data



    def run_with_tasks_old(self):

        # Create task list
        tasks = []
        task: Task = Task(self.get_clients()[0], Client.get_pid)
        task()
        t2 = Task(self.get_clients()[-1], Client.print_pid_and_var, "hello world")
        t2()
        t3 = Task(self.get_clients()[0], Client.train, self.get_clients()[0].get_weights())
        t_c1_join = Task(self.get_server(), Server.client_join, self.get_clients()[0])
        t_c1_train = Task(self.get_clients()[0], Client.train)
        t_c1_update = Task(self.get_server(), Server.client_update, self.get_clients()[0])
        tasks = [t_c1_join]+ [t_c1_train, t_c1_update]*1000
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
    def run_util(cfg):
        sched = Scheduler(**cfg)
        num_rounds = cfg['num_rounds']
        worker_id = int(current_process()._identity[0])
        # print(f'Running task with workerID: {worker_id} <==>')
        # print('')
        # print(f'Starting run with name {cfg["name"]}', end='\r')
        return [sched.run_no_tasks(num_rounds, position=worker_id, add_descr=f'[Worker {worker_id}] '), cfg]

    @staticmethod
    def run_multiple(list_of_configs, pool_size=5):
        outputs = []
        
        pool = Pool(pool_size, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        # pbar = tqdm(total=len(list_of_configs))
        # @TODO: Make sure the memory is dealocated when the task is finished. Currently is accumulating memory with lots of tasks
        outputs = [x for x in tqdm(pool.imap_unordered(Scheduler.run_util, list_of_configs), total=len(list_of_configs), position=0, leave=None, desc='Total')]

        # for i in tqdm(pool.imap_unordered(Scheduler.run_util, list_of_configs, progress_disabled=True)):
        #     print(i)
        # with Pool(pool_size) as p:
        #     outputs = p.map(Scheduler.run_util, list_of_configs)
        return outputs
        # for cfg in list_of_configs:
        #     print(f'Running config: {cfg["name"]}')
        #     num_rounds = cfg['num_rounds']
        #     sched = Scheduler(**cfg)
        #     server_metrics = sched.run_no_tasks(num_rounds)
        #     outputs.append([server_metrics, cfg])
        # return outputs
