from typing import Optional, Tuple, Any, Union, Callable, Iterator, Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader
# from afl.dataset.mnist import load_data
import torch.nn.functional as F
from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, DistributedSampler
from torchvision import datasets, transforms

def afl_dataset(name: str, train_batch_size=128, test_batch_size=128, client_id=0, n_clients=1, seed=-1, use_iter=True) -> Optional[
    Tuple[Iterator[Any], Iterator[Any]]]:
    if name == 'mnist':
        return load_data("~/data", train_batch_size, test_batch_size, client_id, n_clients, seed, use_iter)
    return None


def get_sampler(dataset, n_clients=1, rank=0, seed=-1, shuffle=True) -> SubsetRandomSampler:
    if seed > 0:
        g = torch.Generator()
        g.manual_seed(seed)
        np.random.seed(seed)
    dataset_size = len(dataset)
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    client_indices = indices[rank:dataset_size:n_clients]
    print(f'Sampler for client {rank}')
    return SubsetRandomSampler(client_indices)


def load_data(data_root: str, train_batch_size: int,
              test_batch_size: int,
              client_id: int,
              n_clients: int, seed= -1, use_iter=True) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    # @TODO: Add seed for deterministic sampling
    train_sampler = get_sampler(train_dataset, n_clients, client_id, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler)

    test_sampler = get_sampler(test_dataset, n_clients, client_id, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler)
    if use_iter:
        return iter(train_loader), iter(test_loader)
    return train_loader, test_loader