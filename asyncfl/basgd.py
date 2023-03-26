

import copy
import numpy as np
import torch
from asyncfl import Server
from asyncfl.network import unflatten_g


class Buffer:

    def __init__(self):
        self.avg_gradient = None
        self.N = 0

    def add(self, gradient):
        # print(type(gradient[0]))
        if self.N:
            # tmp_N = ((self.N - 1) / self.N)
            # print(type(((self.N - 1) / self.N)))


            for idx, (avg_g, g) in enumerate(zip(self.avg_gradient, gradient)):
                # Numpy version
                # self.avg_gradient[idx] = ((self.N - 1) / self.N) * avg_g + (1/self.N)*g.copy()
                # Torch version
                self.avg_gradient[idx] = ((self.N - 1) / self.N) * avg_g + (1/self.N)*g.clone()

            #     self.avg_gradient[idx] = ((self.N - 1) / self.N) * self.avg_gradient + (1/self.N)*g.copy()
            # self.avg_gradient = [((self.N - 1) / self.N) * self.avg_gradient + (1/self.N)*g.copy() for g in gradient]
            # self.avg_gradient =  (1/self.N)*gradient
            # self.avg_gradient =  2*gradient
        else:
            self.avg_gradient = gradient
        self.N += 1

    def reset(self):
        self.N = 0
        self.avg_gradient = None

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

    def nonEmpty(self):
        # Check if all the buffers have at least 1 gradient
        for b in self.buffers:
            if not len(b):
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
    pass

    def __init__(self, B):
        self.buffers = []
        self.B = B
        # print(f'Value of B={B}')
        # Create all the buffers
        for b in range(B):
            # print(f'iter b={b}')
            self.buffers.append(Buffer())
        # print('End of buffer init')

    def receive(self, gradient, client_id):
        b = client_id % self.B
        # print(f'Adding gradient from with pid={client_id} to buffer {b}')
        self.buffers[b].add(gradient)

    def nonEmpty(self):
        # Check if all the buffers have at least 1 gradient
        for b in self.buffers:
            if not len(b):
                return False
        return True

    def get_all_gradients(self):
        # print(f'Number of buffers={self.B} and len-> {len(self.buffers)}')
        gradients = [copy.deepcopy(b.avg_gradient) for b in self.buffers]
        for b in self.buffers:
            b.reset()
        # print(f'Buffers are empty? {not self.nonEmpty()}')
        # return gradients
        for x in gradients[0]:
            np.array(x)
        return gradients
        # return np.array(gradients)


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
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, num_buffers, aggr_mode: str = 'async', q=1):
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.buffers = BufferSet(num_buffers)
        self.aggr_mode = aggr_mode
        self.q = q

    def client_update(self, client_id: int, gradients: np.ndarray, client_lipschitz, gradient_age: int):
        grads = torch.from_numpy(gradients)
        # print(f'Got gradient from client {client_id}: grad_age={gradient_age}, server_age={self.get_age()}, diff={self.get_age() - gradient_age}')
        self.buffers.receive(grads, client_id)
        if self.buffers.nonEmpty():
            self.optimizer.zero_grad()
            buffer_gradients = self.buffers.get_all_gradients()
            if self.aggr_mode == 'median':
                avg_grad = median_aggregation(buffer_gradients)
            elif self.aggr_mode == 'trmean':
                avg_grad = trmean_aggregation(buffer_gradients, self.q)
            elif self.aggr_mode == 'krum':
                avg_grad = krum_aggregation(buffer_gradients, self.q)
            else:
                avg_grad = torch.mean(torch.stack(buffer_gradients), dim=0)
            self.aggregate(avg_grad)
        return self.get_model_weights()
