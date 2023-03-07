#
# from .server import Server
from typing import List

from . import Server
from .client import Client
from .task import Task


class Scheduler:

    def __init__(self, dataset_name: str, **config):
        self.dataset_name = dataset_name
        self.clients: List[Client] = []
        self.entities = {}
        self.compute_times = {}
        self.entities['server'] = config['server'](dataset_name, **config['server_args'])
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
        for pid, (c_ct, client_class, client_args) in enumerate(client_data):
            # print(pid, c_ct, client_class, client_args)
            node_id = f'c_{pid}'
            self.entities[node_id] = client_class(pid, self.dataset_name, **client_args)
            self.compute_times[pid] = c_ct

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

    def run_no_tasks(self, num_rounds, ct_clients = []):
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
        print('Running without any tasks wrapper')

        if not ct_clients:
            ct_clients = [1] * len(self.get_clients())

        interaction_sequence = self.compute_interaction_sequence(self.compute_times, num_rounds)
        print('Starting with every client joining to the server --> Get model')

        clients: List[Client] = self.get_clients()
        server = self.get_server()

        initial_weights = server.get_model_weights()
        for c in clients:
            c.set_weights(initial_weights)

        # Play all the server interactions
        for update_id, client_id in enumerate(interaction_sequence):
            # print(f'Training client {client_id}')
            client = clients[client_id]
            client.train()
            server.client_update(client)
            if update_id % 100 == 0:
                # print('Check accuracy')
                out = server.evaluate_accuracy()
                # print(out)
                print(f'[{update_id}/{num_rounds}]\tAccuracy {out[0]}% => loss = {out[1]}')
        print('We can start training and interacting with the server according the compute times')



    def run(self):

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
