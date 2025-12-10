import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from asyncfl.dataloader import afl_dataloader, afl_dataset2


if __name__ == '__main__':
    print('Starting test')

    num_clients = 200
    # pid=0
    # sampler = 'dirichlet'
    # sampler_args = (0.5, 42)
    # dataset_name = 'mnist'
    # batch_size = 50
    # dataset = afl_dataset2(dataset_name, data_type="train")

    # label_data = []

    # for client_id in tqdm(range(num_clients), desc='Counting labels'):
    #     data_loader = afl_dataloader(dataset, use_iter=False, client_id=client_id, n_clients=num_clients, sampler=sampler, sampler_args=sampler_args, train_batch_size=batch_size)
    #     # assert (len(data_loader)*batch_size * num_clients) == len(dataset)
    #     labels = []
    #     for _data, label, in data_loader:
    #         labels.append(label)
        
    #     labels = torch.cat(labels)
    #     counted_labels,bincount = labels.unique(return_counts=True)
    #     counted_labels, bincount = counted_labels.numpy().tolist(), bincount.numpy().tolist()
    #     # print(bincount)
    #     # print(counted_labels)
    #     combined = [[x,y, f'c{client_id}'] for x, y in zip(counted_labels, bincount)]
    #     # print(combined)
    #     label_data += combined
    # df = pd.DataFrame(label_data, columns=['label', 'count', 'client_id'])
    
    # plt.figure()
    # sns.barplot(df, x='label', y='count', hue='client_id')
    # plt.show()
    # plt.figure()
    # sns.kdeplot(df, x='label', hue='client_id')
    # plt.show()

    # sampler = 'nlabels'
    # sampler_args = {'limit': 6, 'seed': 42}
    # dataset_name = 'mnist'
    # batch_size = 50
    # dataset = afl_dataset2(dataset_name, data_type="train")

    # label_data = []

    # for client_id in tqdm(range(num_clients), desc='Counting labels'):
    #     data_loader = afl_dataloader(dataset, use_iter=False, client_id=client_id, n_clients=num_clients, sampler=sampler, sampler_args=sampler_args, train_batch_size=batch_size)
    #     # assert (len(data_loader)*batch_size * num_clients) == len(dataset)
    #     labels = []
    #     for _data, label, in data_loader:
    #         labels.append(label)
        
    #     labels = torch.cat(labels)
    #     counted_labels,bincount = labels.unique(return_counts=True)
    #     counted_labels, bincount = counted_labels.numpy().tolist(), bincount.numpy().tolist()
    #     # print(bincount)
    #     # print(counted_labels)
    #     combined = [[x,y, f'c{client_id}'] for x, y in zip(counted_labels, bincount)]
    #     # print(combined)
    #     label_data += combined
    # df = pd.DataFrame(label_data, columns=['label', 'count', 'client_id'])
    
    # plt.figure()
    # sns.barplot(df, x='label', y='count', hue='client_id')
    # plt.show()
    # plt.figure()
    # sns.kdeplot(df, x='label', hue='client_id')
    # plt.show()

    sampler = 'limitlabel'
    sampler_args = (7,42)
    dataset_name = 'mnist'
    batch_size = 50
    dataset = afl_dataset2(dataset_name, data_type="train")

    label_data = []

    for client_id in tqdm(range(num_clients), desc='Counting labels'):
        data_loader = afl_dataloader(dataset, use_iter=False, client_id=client_id, n_clients=num_clients, sampler=sampler, sampler_args=sampler_args, train_batch_size=batch_size)
        # assert (len(data_loader)*batch_size * num_clients) == len(dataset)
        labels = []
        for _data, label, in data_loader:
            labels.append(label)
        
        labels = torch.cat(labels)
        counted_labels,bincount = labels.unique(return_counts=True)
        counted_labels, bincount = counted_labels.numpy().tolist(), bincount.numpy().tolist()
        # print(bincount)
        # print(counted_labels)
        combined = [[x,y, f'c{client_id}'] for x, y in zip(counted_labels, bincount)]
        # print(combined)
        label_data += combined
    df = pd.DataFrame(label_data, columns=['label', 'count', 'client_id'])

    print(df.groupby('client_id').sum())
    
    plt.figure()
    sns.barplot(df, x='label', y='count', hue='client_id')
    plt.show()
    plt.figure()
    sns.kdeplot(df, x='label', hue='client_id')
    plt.show()
    