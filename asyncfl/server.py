from typing import Any
import numpy as np
import torch
import copy
from .dataloader import afl_dataloader, afl_dataset
from .client import Client
from .network import get_model_by_name, model_gradients, flatten, unflatten_g
import logging

def fed_avg(parameters, sizes = []):
    if not sizes:
        sizes = [1] * len(parameters)
    new_params = {}
    sum_size = 0.0
    for client in parameters:
        for name in parameters[client].keys():
            # try:
            #     new_params[name].data += (parameters[client][name].data * sizes[client])
            # except:
            new_params[name] = (parameters[client][name].detach().clone() * sizes[client])
        sum_size += sizes[client]

    for name in new_params:
        # @TODO: Is .long() really required?
        # logging.info(f'Devide by size of {sum_size} for {new_params[name]}')
        # logging.info(f'Result is: {new_params[name]/ sum_size}')
        new_params[name] = new_params[name] / sum_size
    return new_params




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    # keys = [x for x in update.keys() if 'bn.' not in x]
    keys = [x for x in update.keys()]
    for key in keys:
        # logging.info(f'Get update key {key}')
        update2[key] = update[key].detach().clone() - model[key].detach().clone()
    return update2

def get_update_no_bn(update, model, alpha=0.5):
    '''
    Get the update without the info for the batch normalization parameters

    '''
    pass

def no_defense_update(params, global_parameters, learning_rate=1):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                # logging.info(f'Key: {key}')
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                # logging.info(f'Key: {key}')
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in sum_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += learning_rate*(sum_parameters[var] / total_num)

    return global_parameters

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

class Server:

    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005) -> None:
        self.g_flat = None
        self.clients = []
        self.n = n
        self.f = f
        self.model_history = [] # Indexed by time t
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
        self.prev_weights =  torch.zeros_like(self.w_flat)
        self.prev_gradients = torch.zeros_like(self.w_flat)
        self.prev_prev_gradients = torch.zeros_like(self.w_flat)
        self.age = 0
        self.lips = {}
        self.bft_telemetry = []
        self.model_history.append(self.get_model_weights())
        # self.client_update_history = []
        # self.bft_telemetry = {
        #     "accepted": {
        #         i: {
        #             "values":[],
        #             "total":0
        #         } for i in range(n)
        #     }, 
        #     "rejected" : {
        #         i: {
        #             "values":[],
        #             "total":0
        #         } for i in range(n)
        #     }
        # }

    def set_weights(self, weights):

        keys = weights.keys()
        old_weights = self.network.state_dict()
        for key in keys:
            logging.info(f'[{key}] equal ? {torch.eq(weights[key], old_weights[key])}')
        # logging.info(f'Setting server weights: {weights}')
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

    def incr_age(self):
        self.age += 1

    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool):
        server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
        update_params = get_update(weights, self.model_history[server_model_age])
        client_weight_vec = parameters_dict_to_vector_flt(weights)
        # self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine, client_weight_vec.cpu().numpy().tolist()])
        self.bft_telemetry.append([self.age, client_id, gradient_age, is_byzantine])

        # Aggregate
        # alpha = self.learning_rate / float(gradient_age)
        self.set_weights(no_defense_update([update_params], self.get_model_weights()))
        self.model_history.append(self.get_model_weights())
        self.incr_age()
        return self.get_model_weights()

        

    def client_update(self, client_id: int, gradients: np.ndarray, client_lipschitz, client_convergence, gradient_age: int, is_byzantine: bool):
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
        self.prev_weights = flatten(self.network)
        unflatten_g(self.network, client_gradients, self.device)
        self.optimizer.step()
        self.prev_prev_gradients = self.prev_gradients.clone()
        self.prev_gradients = client_gradients.clone()

    def evaluate_accuracy(self):
        self.network.eval()
        correct = 0
        total = 0
        loss: Any = None

        with torch.no_grad():
            for _batch_idx, (data, target) in enumerate(self.test_set):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.network(data)
                loss = self.network.criterion(outputs, target)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                logging.info(f'Eval --> correct: {correct}, total: {total}')
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
