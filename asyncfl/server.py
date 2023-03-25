import numpy as np
import torch
import copy
from .dataloader import afl_dataloader, afl_dataset
from .client import Client
from .network import get_model_by_name, model_gradients, flatten, unflatten_g


class Server:

    def __init__(self, dataset, model_name: str, learning_rate: float = 0.005) -> None:
        self.g_flat = None
        self.clients = []
        self.test_set = afl_dataloader(
            dataset, use_iter=False, client_id=0, n_clients=1, data_type='test')
        # self.dataset =
        # self.dataset_name = dataset
        # self.test_set = afl_dataset(
        #     self.dataset_name, use_iter=False, client_id=0, n_clients=1, data_type='test')
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.network = get_model_by_name(model_name).to(self.device)
        self.learning_rate = learning_rate
        # @TODO: Set learning rate dynamic
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), lr=self.learning_rate)
        self.w_flat = flatten(self.network)
        self.age = 0

    def set_weights(self, weights):
        self.network.load_state_dict(copy.deepcopy(weights))

    def get_model_weights(self):
        return self.network.state_dict()

    def get_model_gradients(self):
        model_gradients(self.network)

    def client_join(self, client: Client):
        print(f'Client {client.get_pid()} joined the training')
        self.send_model(client)

    def send_model(self, client: Client):
        client.set_weights(self.get_model_weights(), self.get_age())

    def get_age(self):
        return self.age

    def client_update(self, _client_id: int, gradients: np.ndarray, gradient_age: int):
        client_gradients = torch.from_numpy(gradients)
        # print(f'Got gradient from client {_client_id}: grad_age={gradient_age}, server_age={self.get_age()}, diff={self.get_age() - gradient_age}')
        self.aggregate(client_gradients)
        return flatten(self.network)
        # return self.get_model_weights()

    # def client_update(self, client: Client):
    #     # self.g_flat = torch.zeros_like(self.w_flat)
    #     client_gradients = client.get_gradients()
    #     client_gradients = torch.from_numpy(client_gradients)
    #     # Transform flat vector into model dimensions
    #     # unflatten_g(self.network, client_gradients, self.device)
    #     self.aggregate(client_gradients, client.get_pid())
    #     current_model_weights = self.get_model_weights()
    #     client.set_weights(current_model_weights)

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

    def aggregate(self, client_gradients):
        """Merges the new gradients and assigns them to their relevant parameter
        Uses the optimizer update the model based on the gradients
        """
        unflatten_g(self.network, client_gradients, self.device)
        self.optimizer.step()
        self.age += 1

    def evaluate_accuracy(self):
        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _batch_idx, (data, target) in enumerate(self.test_set):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.network(data)
                loss = self.network.criterion(outputs, target)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100. * correct / total, loss.item()

    # def create_clients(self, n, config=None):
    #     compute_times = []
    #     if (not config):
    #         compute_times = [[x, 1] for x in range(n)]
    #     print(compute_times)
    #     for pid, ct in compute_times:
    #         self.clients.append(Client(pid, self.dataset_name))
    #     print(self.clients)

    def run(self):
        print('Running')
