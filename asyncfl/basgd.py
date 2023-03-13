

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

    def get_all_gradients(self) -> np.array:
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

    def get_all_gradients(self) -> np.array:
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


class BASGD(Server):
    def __init__(self, dataset, num_buffers):
        super().__init__(dataset)
        print('Loading BASGD server')
        self.buffers = BufferSet(num_buffers)

        # def aggregate_buffer(self, g_history, q, mode):
        #     # @TODO: Make sure that telerig flattens the gradient in a single vector, not in a list of numpy ndarrays
        #     # Use a torch tensor instead of numpy arrays
        #     if mode == 'median':
        #         length = len(g_history)
        #         half_length = int((length - 1) / 2)
        #         g_list = []
        #         for i in range(len(g_history)):
        #             g_list.append(g_history[i])
        #         g_list, _ = torch.sort(torch.stack(g_list), dim=0)
        #         g = torch.mean(g_list[half_length: length - half_length], dim=0)
        #         return g
        #
        #     elif mode == 'trmean':
        #         length = len(g_history)
        #         g_list = []
        #         for i in range(len(g_history)):
        #             g_list.append(g_history[i])
        #         g_list, _ = torch.sort(torch.stack(g_list), dim=0)
        #         g = torch.mean(g_list[q:length - q], dim=0)
        #         return g

    def aggregate(self, gradient, worker_id=None):
        """Update function for BASGD

        How to rewrite this?
        Input will be a flat numpy vector

        At some point the (aggregated) vector needs to be unflattened and loaded onto the model
        """
        # self.model.version += 1
        grad_data = gradient
        # grad_data = gradient["value"]
        self.buffers.receive(grad_data, worker_id)
        if self.buffers.nonEmpty():
            # print('Aggregate')
            # applying the gradient and performing a step
            self.optimizer.zero_grad()
            # Get the average gradient from the buffers and reset all the buffers
            buffer_gradients = self.buffers.get_all_gradients()


            def avg_gradients(gradients):
                out = []

                for idx, x in enumerate(gradients[0]):
                    out.append([np.mean(x[idx], axis=0) for x in gradients])
                return out
            # avg_grad = avg_gradients(buffer_gradients)
            # avg_grad = torch.mean(buffer_gradients, dim=1)
            avg_grad = torch.mean(torch.stack(buffer_gradients), dim=0)
            # avg_grad = np.mean(buffer_gradients, axis=0)
            unflatten_g(self.network, avg_grad, self.device)
            # avg_grad = [x for x in np.mean(buffer_gradients, axis=0)]

            # def avg_grads(gradients):
            #     avg_grads = []
            #     for idx, _tensor in enumerate(gradients[0]):
            #         avg_grads.append(torch.mean(torck.stac))
            # avg_grad = [x for x in enumerate(buffer_gradients[0])]
            # tstack = torch.stack(buffer_gradients)
            # avg_grad = torch.mean(tstack, dim=0)
            # print(f'Type avg_grad={type(avg_grad)}')
            # print(f'Dim avg_grad={avg_grad}')
            # print(avg_grad)
            #
            # print(f'Type buffer_gradients={type(buffer_gradients)}')
            # print(f'Dim buffer_gradients={buffer_gradients.shape}')
            # print(buffer_gradients)
            # avg_grad = self.buffers.get_all_gradients()
            # self.set_gradients(avg_grad)
            # self.grad_history[self.model.version] = [g.copy() for g in self.model.get_gradients()]
            # self.prev_weights = {key: value.detach().clone() for key, value in self.model.get_weights().items()}
            self.optimizer.step()
        # else:
        #     print('Buffer empty')

        # In any case, return the last known model
        # return self.model.get_weights()