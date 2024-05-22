import logging
from pathlib import Path
from typing import Any, Iterator, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchvision import datasets, transforms

from asyncfl.datasampler import (DirichletSampler, LimitLabelsSampler,
                                 LimitLabelsSamplerFlex, N_Labels, get_sampler,
                                 uniform_sampler_func)

# 400
# def afl_dataset(name: str, train_batch_size=128, test_batch_size=128, client_id=0, n_clients=1, seed=-1, use_iter=True, data_root: str = '~/data', sampler='uniform', sampler_args={}) -> Optional[


def afl_dataset2(
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
):
    if name == "mnist":
        data_set = load_mnist(data_root, data_type=data_type)
    elif name == "cifar10":
        data_set = load_cifar10(data_root, data_type=data_type)
    elif name == "cifar100":
        data_set = load_cifar100(data_root, data_type=data_type)
    elif name == "wikitext2":
        data_set = load_wikitext2(data_root, data_type=data_type)
    else:
        raise ValueError(f'Unknown dataset name "{name}"!')
    return data_set


def afl_dataloader(dataset, train_batch_size=50,
    test_batch_size=400,
    client_id=0,
    n_clients=1,
    seed=-1,
    data_type: str = "train",
    use_iter=True,
    sampler="uniform",
    sampler_args={},
    drop_last=False
    ) -> DataLoader:

    # Print the number of clients
    # print(f'[AFL DL] ({client_id}) Number of clients: {n_clients}')

    if sampler == 'dirichlet':
        logging.info('Using dirichlet sampler')
        ds = DirichletSampler(dataset, n_clients, client_id, (sampler_args))
        indices = ds.indices
    elif sampler == 'nlabels':
        logging.info('Using nlabels sampler')
        ds = N_Labels(dataset, n_clients, client_id, seed=seed, **sampler_args)
        indices = ds.indices
    elif sampler == 'limitlabelflex':
        ds = LimitLabelsSamplerFlex(dataset, n_clients, client_id, (sampler_args))
        indices = ds.indices
    elif sampler == 'limitlabel':
        ds = LimitLabelsSampler(dataset, n_clients, client_id, (sampler_args))
        indices = ds.indices
    else:
        logging.info('Using uniform sampler')
        indices = uniform_sampler_func(dataset, n_clients, client_id, seed, **sampler_args)

    ds_subset = Subset(dataset, indices)
    if data_type == "train":
        train_loader = DataLoader(
            ds_subset, batch_size=train_batch_size, shuffle=True, drop_last=drop_last
        )
        if use_iter:
            return iter(train_loader)
        return train_loader
    else:
        test_loader = DataLoader(
            ds_subset, batch_size=test_batch_size, shuffle=True, drop_last=drop_last
        )
        if use_iter:
            return iter(test_loader)
        return test_loader

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

    # sampler_class = get_sampler(sampler)
    # data_sampler = sampler_class(
    #         data_set, n_clients, client_id, seed, **sampler_args
    #     )
    # return None

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
    data_set = datasets.CIFAR10(data_root, train=True, download=True, transform=None)
    indices = list(range(0, len(data_set), n_clients))

    ds_subset = Subset(data_set, indices)
    # data_set = None
    # print('fINIHSED')
    # return None
    if data_type == "train":
        # train_sampler = sampler_class(
        #     data_set, n_clients, client_id, seed, **sampler_args
        # )
        # train_loader = DataLoader(
        #     data_set, batch_size=train_batch_size, sampler=train_sampler
        # )
        train_loader = DataLoader(
            ds_subset, batch_size=train_batch_size, shuffle=True
        )
        if use_iter:
            return iter(train_loader)
        return train_loader
    else:
        # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        # test_sampler = sampler_class(
        #     data_set, n_clients, client_id, seed, **sampler_args
        # )
        # test_loader = DataLoader(
        #     data_set, batch_size=test_batch_size, sampler=test_sampler
        # )
        test_loader = DataLoader(
            ds_subset, batch_size=test_batch_size, shuffle=True
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
        return datasets.CIFAR10(data_root, train=True, download=True, transform=transform
        )
    else:
        return datasets.CIFAR10(data_root, train=False, download=False, transform=transform)


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
            data_root, train=True, download=True, transform=transform
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


class WikiText2Dataset(Dataset):
    def __init__(self, tokenized_data: list, vocab: Vocab, block_size: int) -> None:
        transformed_data = [self.transform_text(x, vocab) for x in tokenized_data]
        self.data, self.target = self.group_texts(transformed_data, block_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx, :], self.target[idx, :])

    @staticmethod
    def transform_text(text, vocab):
        if text:
            mapped = [vocab[token] for token in text]
            mapped.append(vocab["<eos>"])
            return torch.tensor(mapped, dtype=torch.long)
        return torch.tensor([], dtype=torch.long)

    @staticmethod
    def group_texts(texts, block_size):
        concatenated_examples = torch.cat(texts)
        nr_items = concatenated_examples.numel() // block_size
        total_length = nr_items * block_size
        res_data = concatenated_examples[:total_length].view(nr_items, block_size)
        res_target = concatenated_examples[1:total_length + 1].view(nr_items, block_size)

        return res_data, res_target


def load_wikitext2(data_root: str, data_type: str):
    data_root_expanded_user = str(Path(data_root).expanduser())
    ds_train = WikiText2(root=data_root_expanded_user, split="train")
    tokenizer = get_tokenizer('basic_english')
    tokenized_ds_train = [tokenizer(x) for x in ds_train]
    vocab = build_vocab_from_iterator(
        tokenized_ds_train, specials=["<unk>", "<eos>", "<pad>"], min_freq=4)
    vocab.set_default_index(0)

    if data_type == "train":
        return WikiText2Dataset(tokenized_ds_train, vocab, 128)
    else:
        ds_test = WikiText2(root=data_root_expanded_user, split="test")
        tokenized_ds_test = [tokenizer(x) for x in ds_test]
        return WikiText2Dataset(tokenized_ds_test, vocab, 128)
