import copy
import logging
from math import exp
from typing import Any, List, Tuple, cast

import numpy as np
import torch

from asyncfl.backdoor_util import add_trigger, save_img, test_or_not
from asyncfl.scheduler_util import SchedulerContext

from .client import Client
from .dataloader import afl_dataloader, afl_dataset
from .network import (TextLSTM, flatten, flatten_dict, get_model_by_name,
                      model_gradients, unflatten_dict, unflatten_g)


        
def fed_avg_vec(params: List[np.ndarray]) -> np.ndarray:
    '''
    Use to average the clients weights when representation a N 1 dimensional numpy vectors
    '''
    logging.info('Running (sync) fed average vector version')
    weigths = np.ones(len(params))
    averaged: np.ndarray = np.average(params, axis=0, weights=weigths)
    return averaged

# @TODO: Rewrite this to make it work with numpy arrays?
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

def no_defense_vec_update(params: List[np.ndarray], global_params: np.ndarray, server_rl = 1.0) -> np.ndarray:
    summed: np.ndarray = np.sum(params, axis=0)
    # logging.info(f'Shape of summed = {summed.shape}')
    # logging.info(f'Shape of global params = {global_params.shape}')
    return global_params + server_rl * (summed / float(len(params)))

def no_defense_update(params, global_parameters, learning_rate=1.0):
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
    assert type(sum_parameters) == dict
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

    def __init__(self, n, f, dataset, model_name: str, learning_rate: float = 0.005, backdoor_args = {}) -> None:
        self.g_flat = None
        self.clients = []
        self.n = n
        self.f = f
        self.model_history = [] # Indexed by time t
        self.model_client_history = {}
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
        self.sched_ctx : SchedulerContext
        self.is_lstm = isinstance(self.network, TextLSTM)
        self.test_set = afl_dataloader(
            dataset, test_batch_size=100 if self.is_lstm else 400,
            use_iter=False, client_id=0, n_clients=1, data_type='test',
            drop_last=self.is_lstm)

        # Updated way
        # self.model_history.append(self.get_model_dict_vector())
        for i in range(n):
            self.model_client_history[i]=self.get_model_dict_vector()
        self.test_backdoor = False
        if backdoor_args != {}:
            self.test_backdoor = True
            self.backdoor_args = backdoor_args



        # Old way
        # self.model_history.append(self.get_model_weights())


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
        # for key in keys:
        #     logging.info(f'[{key}] equal ? {torch.eq(weights[key], old_weights[key])}')
        # logging.info(f'Setting server weights: {weights}')
        self.network.load_state_dict(copy.deepcopy(weights))

    def get_model_weights(self):
        return self.network.state_dict()

    def get_model_dict_vector(self) -> np.ndarray:
        return flatten_dict(self.network).cpu().numpy().copy()

    def load_model_dict_vector_t(self, vec: torch.Tensor):
        unflatten_dict(self.network, vec)

    def load_model_dict_vector(self, vec: np.ndarray):
        tensor_vec = torch.from_numpy(vec).to(self.device)
        unflatten_dict(self.network, tensor_vec)

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


    def aggregate_sync(self, params: List[np.ndarray], byz_clients) -> np.ndarray:
        '''
        Use to average the clients weights when representation a N 1 dimensional numpy vectors
        '''
        logging.info('[Server] Running (sync) fed average vector version')
        weigths = np.ones(len(params))
        averaged: np.ndarray = np.average(params, axis=0, weights=weigths)

        self.load_model_dict_vector(averaged)
        self.incr_age()
        return averaged.copy()
    
    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> Tuple[np.ndarray, bool]:
        logging.info(f'Default server dict_vector update of client {client_id}')
        # server_model_age = gradient_age if gradient_age < len(self.model_history) else 0
        server_model = self.model_client_history[client_id]
        # model_difference = weight_vec - self.model_history[server_model_age] # Gradient approximation
        model_difference = weight_vec - server_model # Gradient approximation
        updated_model_vec = no_defense_vec_update([model_difference], self.get_model_dict_vector(), server_rl=self.learning_rate)
        # self.model_history.append(updated_model_vec)
        self.model_client_history[client_id] = updated_model_vec
        self.load_model_dict_vector(updated_model_vec)
        self.incr_age()
        # logging.info(updated_model_vec)
        return updated_model_vec.copy(), True




    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool):
        logging.info('Default server client weight update')
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
        raise DeprecationWarning(f"Function '{__name__}' is deprecated")
        logging.info('Default server client update')
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
        raise DeprecationWarning(f"Function '{__name__}' is deprecated")
        self.prev_weights = flatten(self.network)
        unflatten_g(self.network, client_gradients, self.device)
        self.optimizer.step()
        self.prev_prev_gradients = self.prev_gradients.clone()
        self.prev_gradients = client_gradients.clone()

    def evaluate_model(self, test_backdoor=False):
        self.network.eval()
        correct = 0
        correct_backdoor = 0
        back_accu = 0
        back_num = 0
        total = 0
        loss: Any = None

        if self.is_lstm:
            hidden = cast(TextLSTM, self.network).init_hidden(
                self.test_set.batch_size, self.device)
        else:
            hidden = None

        loss = 0.

        with torch.no_grad():
            for _batch_idx, (inputs, targets) in enumerate(self.test_set):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if hidden:
                    hidden = cast(TextLSTM, self.network).detach_hidden(hidden)
                    outputs, hidden = self.network(inputs, hidden)
                    outputs = outputs.reshape(inputs.numel(), -1)
                    targets = targets.reshape(-1)
                    loss += self.network.criterion(outputs, targets, reduction="sum").item()
                else:
                    outputs = self.network(inputs)
                    loss += self.network.criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if self.test_backdoor:
                    assert self.backdoor_args 
                    # logging.debug(f'{self.backdoor_args=}')
                    # logging.debug('[SERVER BACKDOOR EVAL].....')
                    del_arr = []
                    for k, image in enumerate(inputs):
                        # logging.debug(f'{k=}, {len(targets)=}, {len(inputs)=}')
                        # label_val = int(targets_aux[k].cpu().numpy().tolist())
                        label_val = 4
                        # logging.debug(f'{label_val=}')
                        # logging.debug(f'{targets_aux.device=}')
                        # logging.debug(f'{targets.device=}')
                        if test_or_not(self.backdoor_args, targets[k]):  # one2one need test
                            # inputs[k][:, 0:5, 0:5] = torch.max(inputs[k])
                            inputs[k] = add_trigger(self.backdoor_args,inputs[k], self.device)
                            save_img(inputs[k])
                            # logging.debug(f'[BACK DEBUG] changing label from {targets[k]} to { int(self.backdoor_args["attack_label"])}')
                            targets[k] = int(self.backdoor_args['attack_label'])
                            back_num += 1
                        else:
                            # logging.debug(f'Keeping the label at {targets[k]}')
                            pass
                            targets[k] = int(-1)
                    outputs = self.network(inputs)
                    # loss += self.network.criterion(outputs, targets).item()
                    # log_probs = net_g(data)
                    predicted = outputs.data.max(1, keepdim=True)[1]
                    # _, predicted = torch.max(outputs.data, 1)
                    correct_backdoor += predicted.eq(targets.data.view_as(predicted)).long().cpu().sum().item()
                    # correct_backdoor += (predicted == targets).sum().item()
        if self.test_backdoor:
            back_accu = float(correct_backdoor) / back_num
        logging.info(f'Eval --> correct: {correct}, total: {total}')
        logging.info(f'Eval --> correct_backdoor: {correct_backdoor} ({back_num}), total: {total}')

        loss /= total

        return exp(loss) if hidden else 100. * correct / total, loss, back_accu

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
