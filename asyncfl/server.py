import numpy as np
import torch

from .dataloader import afl_dataset
from .client import Client
from .network import MNIST_CNN, get_model_by_name, model_gradients, flatten, unflatten_g


class Server:


    def __init__(self, dataset: str, model_name: str) -> None:
        self.g_flat = None
        # print('Hello server')
        self.clients = []
        self.dataset_name = dataset
        self.train_set, self.test_set = afl_dataset(self.dataset_name, use_iter=False, client_id=0, n_clients=1)
        # if torch.cuda.is_available()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        # self.network = MNIST_CNN().to(self.device)
        self.network = get_model_by_name(model_name).to(self.device)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)
        self.w_flat = flatten(self.network)




    def get_model_weights(self):
        return self.network.state_dict()

    def get_model_gradients(self):
        model_gradients(self.network)

    # def client_update(self, weights, pid):
    #     return None

    def client_join(self, client: Client):
        print(f'Client {client.get_pid()} joined the training')
        self.send_model(client)

    def send_model(self, client: Client):
        client.set_weights(self.get_model_weights())

    def client_update(self, client: Client):
        # w_flat = flatten(self.network)
        self.g_flat = torch.zeros_like(self.w_flat)
        client_gradients = client.get_gradients()
        client_gradients = torch.from_numpy(client_gradients)
        unflatten_g(self.network, client_gradients, self.device)
        # print(f'Received update from {type(client)} with PID: {client.get_pid()}')
        self.aggregate(client_gradients, client.get_pid())
        current_model_weights = self.get_model_weights()
        client.set_weights(current_model_weights)

    def set_gradients(self, gradients):
        """Merges the new gradients and assigns them to their relevant parameter"""
        for g, p in zip(gradients, self.network.parameters()):
            if g is not None:
                try:
                    # grad = torch.from_numpy(np.array(g, dtype=np.float16))
                    p.grad = torch.from_numpy(g).to(self.device)
                except Exception as e:
                    print('Exception')
                    print(g)
                    raise e
        # self.network.to(self.device)

    def aggregate(self, client_gradients, client_id):
        """Merges the new gradients and assigns them to their relevant parameter
        Uses the optimizer update the model based on the gradients
        @TODO: Want to replace this with a manual aggregation step


        How to rewrite this?
        Input will be a flat numpy vector?
        """
        unflatten_g(self.network, client_gradients, self.device)
        # print(client_gradients)
        # for g, p in zip(client_gradients, self.network.parameters()):
        #     if g is not None:
        #         try:
        #             # p.grad = torch.from_numpy(np.array(g, dtype=np.float16))
        #             # p.grad = torch.from_numpy(g).to(self.device)
        #             p.grad = g.to(self.device)
        #             # p.grad = g
        #         except Exception as e:
        #             print('Exception')
        #             print(g)
        #             raise e
        self.optimizer.step()
        # for g in client_gradients:
        #     g_flat = torch.zeros_like(g)
        #     u_flat = torch.zeros_like(g)
        # print('Aggregating')
    # def init_network(self):


    def evaluate_accuracy(self):
        # print('Testing')

        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_set):
                data, target = data.to(self.device), target.to(self.device)
                # This is only set to finish evaluation faster.
                # if batch_idx * len(data) > 1024:
                #     break
                outputs = self.network(data)
                loss = self.network.criterion(outputs, target)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        # print('inter')
        return 100. * correct / total, loss.item()
    
    def create_clients(self, n, config = None):
        compute_times = []
        if(not config):
            compute_times = [[x, 1] for x in range(n)]
        print(compute_times)
        for pid, ct in compute_times:
            self.clients.append(Client(pid, self.dataset_name))
        print(self.clients)

    
    def run(self):
        print('Running')