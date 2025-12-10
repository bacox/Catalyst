

import copy
import inspect
from typing import List
import numpy as np
import torch
from asyncfl import Server
from asyncfl.network import flatten, unflatten_g
import sys
import logging

class Buffer:

    def __init__(self):
        self.avg_gradient = None
        self.N = 0
        self.client_ids = []

    def add(self, gradient, client_id: int):
        # print(type(gradient[0]))
        if client_id in self.client_ids:
            logging.warning(f'Not adding client {client_id} to buffer. Already there')
            return
        if self.N > 0:
            # tmp_N = ((self.N - 1) / self.N)
            # print(type(((self.N - 1) / self.N)))
            self.avg_gradient = ((self.N - 1) / self.N) * self.avg_gradient + (1/self.N)*gradient.detach().clone()

            # for idx, (avg_g, g) in enumerate(zip(self.avg_gradient, gradient)):
            #     # Numpy version
            #     # self.avg_gradient[idx] = ((self.N - 1) / self.N) * avg_g + (1/self.N)*g.copy()
            #     # Torch version
            #     self.avg_gradient[idx] = ((self.N - 1) / self.N) * avg_g + (1/self.N)*g.clone()

            #     self.avg_gradient[idx] = ((self.N - 1) / self.N) * self.avg_gradient + (1/self.N)*g.copy()
            # self.avg_gradient = [((self.N - 1) / self.N) * self.avg_gradient + (1/self.N)*g.copy() for g in gradient]
            # self.avg_gradient =  (1/self.N)*gradient
            # self.avg_gradient =  2*gradient
        else:
            self.avg_gradient = gradient.detach().clone()
        self.N += 1
        self.client_ids.append(client_id)

    def reset(self):
        self.N = 0
        self.avg_gradient = None
        self.client_ids = []

    def __len__(self):
        return self.N

class BufferSet_G:

    def __init__(self, B):
        self.buffers = []
        self.B = B

        for _b in range(B):
            self.buffers.append(Buffer())

    def receive(self, gradient, client_id):
        b = client_id % self.B
        self.buffers[b].add(gradient)

    def nonEmptyCount(self):
        return sum([1 for x in self.buffers if len(x) > 0 ])
    
    def nonEmpty(self):
        # Check if all the buffers have at least 1 gradient
        for b in self.buffers:
            if len(b) < 1:
                return False
        return True

    def get_all_gradients(self):
        gradients = [copy.deepcopy(b.avg_gradient) for b in self.buffers]
        for b in self.buffers:
            b.reset()

        for x in gradients[0]:
            np.array(x)
        return gradients

class BufferSet:

    def __init__(self, B):
        self.buffers: List[Buffer] = []
        self.B = B
        # print(f'Value of B={B}')
        # Create all the buffers
        for b in range(B):
            # print(f'iter b={b}')
            self.buffers.append(Buffer())
        # print('End of buffer init')

    def receive(self, gradient, client_id):
        b = client_id % self.B
        # print(f'Receive new gradient in buffer {client_id % self.B}')
        # print(f'Adding gradient from with pid={client_id} to buffer {b}')
        self.buffers[b].add(gradient, client_id)
        # print('End receive')

    def nonEmptyCount(self):
        return sum([1 for x in self.buffers if len(x) > 0 ])

    def nonEmpty(self):
        # Check if all the buffers have at least 2 gradients
        for idx,b in enumerate(self.buffers):
            if len(b) < 1:
                # print(f'buffer {idx} is empty\n')
                return False
        return True

    def get_all_gradients(self):
        gradients = [x.avg_gradient for x in self.buffers]
        for b in self.buffers:
            b.reset()
        return gradients
    
    def __len__(self) -> int:
        return sum([len(x) for x in self.buffers])


def median_aggregation(list_of_gradients):
    length = len(list_of_gradients)
    half_length = int((length - 1) / 2)
    g_list = []
    for i in range(len(list_of_gradients)):
        g_list.append(list_of_gradients[i])
    g_list, _ = torch.sort(torch.stack(g_list), dim=0)
    g = torch.mean(g_list[half_length: length-half_length], dim=0)
    return g

def trmean_aggregation(list_of_gradients, q):
    length = len(list_of_gradients)
    g_list = []
    for i in range(len(list_of_gradients)):
        g_list.append(list_of_gradients[i])
    g_list, _ = torch.sort(torch.stack(g_list), dim=0)
    g = torch.mean(g_list[q:length-q], dim=0)
    return g

def krum_aggregation(list_of_gradients, q):
    length = len(list_of_gradients)
    # compute distance matrix
    distance = []
    for i in range(length):
        distance.append(torch.FloatTensor([0]*length))
    for i in range(length):
        for j in range(i+1, length):
            distance_ij = torch.norm(list_of_gradients[i].sub(list_of_gradients[j]), 2)
            distance_ij.mul_(distance_ij)
            distance[i][j] = distance_ij
            distance[j][i] = distance_ij
    # compute krum score
    score = [0] * length
    for i in range(length):
        score[i] = torch.sum(torch.topk(distance[i], length-q-2, 0, largest=False, sorted=False)[0])
    i_star = torch.topk(torch.FloatTensor(score), 1, 0, largest=False, sorted=False)[1]
    return list_of_gradients[i_star]

class BASGD(Server):
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, backdoor_args = {}, num_buffers=2, aggr_mode: str = 'async', q=1):
        super().__init__(n, f, dataset, model_name, learning_rate, backdoor_args)
        self.buffers = BufferSet(num_buffers)
        self.aggr_mode = aggr_mode
        self.q = q

    def client_weight_dict_vec_update(self, client_id: int, weight_vec: np.ndarray, gradient_age: int, is_byzantine: bool) -> np.ndarray:
        has_aggregated = False
        logging.info(f'BaSGD dict_vector update of client {client_id}')
        vec_t = torch.from_numpy(weight_vec).to(self.device)
        logging.info(f'{self.age=}, {gradient_age=}')
        # Block updates that are too old?
        if (self.age - gradient_age) > 0:
            logging.debug('Blocking old update')  
        else:
            self.buffers.receive(vec_t, client_id)
            if self.buffers.nonEmpty():
                logging.info(f'Aggregate! {len(self.buffers)} buffers and {self.aggr_mode=}')
                self.optimizer.zero_grad()
                buffer_gradients = self.buffers.get_all_gradients()
                if self.aggr_mode == 'median':
                    logging.info('agg Median')
                    avg_weight_vec = median_aggregation(buffer_gradients)
                elif self.aggr_mode == 'trmean':
                    avg_weight_vec = trmean_aggregation(buffer_gradients, self.q)
                elif self.aggr_mode == 'krum':
                    avg_weight_vec = krum_aggregation(buffer_gradients, self.q)
                else:
                    # print('Agg mean')
                    avg_weight_vec = torch.mean(torch.stack(buffer_gradients), dim=0)
                has_aggregated = True
                # print(f'Aggregate!!!! --> {avg_weight_vec}')
                # self.aggregate(avg_weight_vec)
                # self.model_history.append(avg_weight_vec)
                self.load_model_dict_vector_t(avg_weight_vec)
                self.incr_age()
            else:
                logging.debug(f'[BASGD] need {len(self.buffers.buffers) - self.buffers.nonEmptyCount()} more buffers')
        # logging.info(updated_model_vec)
        return self.get_model_dict_vector(), has_aggregated


    def client_weight_update(self, client_id, weights: dict, gradient_age: int, is_byzantine: bool):
        raise NotImplementedError(f"Function '{inspect.currentframe().f_code.co_name}' is not implemented yet in {__class__.__name__}")
        return super().client_weight_update(client_id, weights, gradient_age, is_byzantine)

    def client_update(self, client_id: int, gradients: np.ndarray, client_lipschitz, client_convergence, gradient_age: int, is_byzantine: bool):
        raise DeprecationWarning(f"Function '{__name__}' is deprecated")
        grads = torch.from_numpy(gradients)
        # print(f'Got gradient from client {client_id}: grad_age={gradient_age}, server_age={self.get_age()}, diff={self.get_age() - gradient_age}')
        # print('Hang here?')
        self.buffers.receive(grads, client_id)
        if self.buffers.nonEmpty():
            # print(f'Aggregate! {len(self.buffers)} buffers')
            self.optimizer.zero_grad()
            buffer_gradients = self.buffers.get_all_gradients()
            if self.aggr_mode == 'median':
                print('agg Median')
                avg_grad = median_aggregation(buffer_gradients)
            elif self.aggr_mode == 'trmean':
                avg_grad = trmean_aggregation(buffer_gradients, self.q)
            elif self.aggr_mode == 'krum':
                avg_grad = krum_aggregation(buffer_gradients, self.q)
            else:
                # print('Agg mean')
                avg_grad = torch.mean(torch.stack(buffer_gradients), dim=0)
            # print(f'Aggregate!!!! --> {avg_grad}')
            # self.aggregate(avg_grad)
            self.aggregate(avg_grad)
        # print(f'Grads in bufferset: {len(self.buffers)}')
            

        return flatten(self.network)
        # return self.get_model_weights()
