from typing import Optional, Tuple, Any, Iterator, Union
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from asyncfl.datasampler import N_Labels, UniformSampler, get_sampler

# 400
# def afl_dataset(name: str, train_batch_size=128, test_batch_size=128, client_id=0, n_clients=1, seed=-1, use_iter=True, data_root: str = '~/data', sampler='uniform', sampler_args={}) -> Optional[


def afl_dataset(
    name: str,
    train_batch_size=400,
    test_batch_size=400,
    client_id=0,
    n_clients=1,
    seed=-1,
    data_type: str = "train",
    use_iter=True,
    data_root: str = "~/data",
    sampler="uniform",
    sampler_args={},
) -> Union[Iterator[Any], DataLoader[Any]]:
    if name == "mnist":
        data_set = load_mnist(data_root, data_type=data_type)
    elif name == "cifar10":
        data_set = load_cifar10(data_root, data_type=data_type)
    elif name == "cifar100":
        data_set = load_cifar100(data_root, data_type=data_type)
    else:
        raise ValueError(f'Unknown dataset name "{name}"!')
    # @TODO: Add seed for deterministic sampling

    sampler_class = get_sampler(sampler)

    if data_type == "train":
        train_sampler = sampler_class(
            data_set, n_clients, client_id, seed, **sampler_args
        )
        train_loader = DataLoader(
            data_set, batch_size=train_batch_size, sampler=train_sampler
        )
        if use_iter:
            return iter(train_loader)
        return train_loader
    else:
        # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_sampler = sampler_class(
            data_set, n_clients, client_id, seed, **sampler_args
        )
        test_loader = DataLoader(
            data_set, batch_size=test_batch_size, sampler=test_sampler
        )
        # test_loader = DataLoader(data_set, batch_size=test_batch_size)
        if use_iter:
            return iter(test_loader)
        return test_loader


def load_cifar10(data_root: str, data_type: str):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if data_type == 'train':
        return datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform
        )
    else:
        return datasets.CIFAR10(data_root, train=False, transform=transform)


def load_cifar100(data_root: str, data_type: str):
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
    )
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if data_type == 'train':
        return datasets.CIFAR100(
            data_root, train=True, download=False, transform=transform
        )
    else:
        return datasets.CIFAR100(
            data_root, train=False, download=False, transform=transform
        )
    # return train_dataset, test_dataset


def load_mnist(data_root: str, data_type: str):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if data_type == 'train':
        return datasets.MNIST(
            data_root, train=True, download=True, transform=transform
        )
    else:
        return datasets.MNIST(data_root, train=False, transform=transform)
    # return train_dataset, test_dataset
