import random
from torch.utils.data import DistributedSampler, Dataset
from typing import Iterator
import numpy as np
import math

def get_sampler(name:str):
    if name == 'uniform':
        return UniformSampler
    elif name == 'n_labels':
        return N_Labels
    else:
        raise ValueError(f'Unknown Sampler name "{name}"!')

class DistributedSamplerWrapper(DistributedSampler):
    indices = []
    epoch_size = 1.0
    def __init__(self, dataset: Dataset, num_replicas = None,
                 rank = 0, seed = 0) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.client_id = max(0,rank - 1)
        # self.n_clients = num_replicas - 1
        self.n_clients = num_replicas
        if hasattr(self.dataset, 'classes'):
            self.n_labels = len(self.dataset.classes) # type: ignore
        else:
            self.n_labels = 0
        self.seed = seed


    def order_by_label(self, dataset):
        # order the indices by label
        ordered_by_label = [[] for i in range(len(dataset.classes))]
        for index, target in enumerate(dataset.targets):
            ordered_by_label[target].append(index)

        return ordered_by_label

    def set_epoch_size(self, epoch_size: float) -> None:
        """ Sets the epoch size as relative to the local amount of data.
        1.5 will result in the __iter__ function returning the available
        indices with half appearing twice.

        Args:
            epoch_size (float): relative size of epoch
        """
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterator[int]:
        random.seed(self.rank+self.epoch)
        epochs_todo = self.epoch_size
        indices = []
        while(epochs_todo > 0.0):
            random.shuffle(self.indices)
            if epochs_todo >= 1.0:
                indices.extend(self.indices)
            else:
                end_index = int(round(len(self.indices)*epochs_todo))
                indices.extend(self.indices[:end_index])

            epochs_todo = epochs_todo - 1

        ratio = len(indices)/float(len(self.indices))
        np.testing.assert_almost_equal(ratio, self.epoch_size, decimal=2)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.indices)
    

def uniform_sampler_func(dataset: Dataset, num_replicas=1, rank=1, seed=0):
    num_samples = math.ceil(len(dataset) / num_replicas)  # type: ignore[arg-type]
    total_size = num_samples * num_replicas
    indices = list(range(len(dataset))) # type: ignore
    # random.shuffle(indices)
    # print('Func')
    return indices[rank:total_size:num_replicas]


class UniformSampler(DistributedSamplerWrapper):
    def __init__(self, dataset, num_replicas=None, rank=0, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)

        indices = list(range(len(self.dataset))) # type: ignore
        random.shuffle(self.indices)

        self.indices = indices[self.rank:self.total_size:self.n_clients]
        print(f'Indices for rank {rank}:{num_replicas} is of size {len(self.indices)}')


class N_Labels(DistributedSamplerWrapper):
    """
    A sampler that limits the number of labels per client
    The number of clients must <= than number of labels
    """

    def __init__(self, dataset, num_replicas, rank, seed, limit=10):
        super().__init__(dataset, num_replicas, rank, seed)

        num_copies = np.ceil((limit * self.n_clients) / self.n_labels)
        label_dict = {}
        for l in range(self.n_labels):
            label_dict[l] = num_copies
        clients = list(range(self.n_clients))  # keeps track of which clients should still be given a label
        client_label_dict = {}
        ordered_list = list(range(self.n_labels)) * int(num_copies)
        # Now code
        for idx, client_id in enumerate(clients):
            label_set = []
            for _ in range(limit):
                label_set.append(ordered_list.pop())
            client_label_dict[client_id] = label_set

        client_label_dict['rest'] = []
        # New code
        if len(ordered_list):
            client_label_dict['rest'] = ordered_list
        reverse_label_dict = {}
        for l in range(self.n_labels):
            reverse_label_dict[l] = []

        for k, v in client_label_dict.items():
            for l_c in v:
                reverse_label_dict[l_c].append(k)

        indices = []
        ordered_by_label = self.order_by_label(dataset)
        indices_per_client = {}
        for c in clients:
            indices_per_client[c] = []

        rest_indices = []
        for group, label_list in enumerate(ordered_by_label):
            splitted = np.array_split(label_list, num_copies)
            client_id_to_distribute = reverse_label_dict[group]
            for split_part in splitted:
                client_key = client_id_to_distribute.pop()
                if client_key == 'rest':
                    rest_indices.append(split_part)
                else:
                    indices_per_client[client_key].append(split_part)
        # @TODO: Fix this part in terms of code cleanness. Could be written more cleanly
        if len(rest_indices):
            rest_indices = np.concatenate(rest_indices)
            rest_splitted = np.array_split(rest_indices, len(indices_per_client))

            for k, v in indices_per_client.items():
                v.append(rest_splitted.pop())
                indices_per_client[k] = np.concatenate(v)
        else:
            rest_indices = np.ndarray([])
            for k, v in indices_per_client.items():
                indices_per_client[k] = np.concatenate(v)

        indices = indices_per_client[self.client_id]
        random.seed(seed + self.client_id)  # give each client a unique shuffle
        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices