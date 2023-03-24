import argparse
from pathlib import Path
import numpy as np
import json
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
# Turn interactive plotting off
plt.ioff()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "-omit", help="Omit running experiment",
                        action="store_true")
    parser.add_argument('-a', '--autocomplete',
                        help="Autocomplete missing experiments. Based on the results in the datafile, missing experiment will be run.", action='store_true')
    args = parser.parse_args()

    print('Exp 07: Staleness Aware SGD param sweep')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp07_sa_sgd'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        # Define configurations
        configs = []
        model_list = ['cifar10-lenet']
        dataset = 'cifar10'
        f = 0  # number of byzantine clients
        # num_rounds = 50*10
        num_rounds = 500
        idx = 1
        repetitions = 1
        limit = 10
        # num_clients = [50]
        num_clients = [25]
        exp_id = 0
        # num_clients = [50, 25, 10, 5, 1]
        for _r in range(repetitions):
            for n in num_clients:
                for model_name in model_list:
                    exp_id += 1
                    configs.append({
                        'exp_id': exp_id,
                        'aggregation_type': 'async',
                        'name': f'sasgd-{model_name}-{dataset}-n{n}_async',
                        'num_rounds': num_rounds,
                        'clients': {
                                'client': AFL.Client,
                                'client_args': {
                                    'sampler': 'uniform',
                                    'sampler_args': {
                                    }
                                },
                            'client_ct': [1] * (n - f),
                            'n': n,
                            'f': f,
                            'f_type': AFL.NGClient,
                            'f_args': {'magnitude': 10},
                            'f_ct': [1] * f
                        },
                        'server': AFL.SaSGD,
                        'server_args': {
                        },
                        'dataset_name': dataset,
                        'model_name': model_name
                    })

        # Run all experiments multithreaded
        completed_runs = []
        if args.autocomplete:
            print('Running mising experiments:')
            with open(data_file, 'r') as f:
                completed_runs = json.load(f)
                keys = [x[1]['exp_id'] for x in completed_runs]
                configs = [x for x in configs if x['exp_id'] not in keys]
                # @TODO: Append to output instead of overwriting
                # print(configs)
            # exit()

        

        import torchvision
        import torch

        trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                                download=True, transform=None)

        odds1 = list(range(0, len(trainset), 5))
        odds2 = list(range(1, len(trainset), 5))
        odds3 = list(range(2, len(trainset), 5))
        odds4 = list(range(3, len(trainset), 5))
        odds5 = list(range(4, len(trainset), 5))
        trainset_1 = torch.utils.data.Subset(trainset, odds1)
        trainset_2 = torch.utils.data.Subset(trainset, odds2)
        trainset_3 = torch.utils.data.Subset(trainset, odds3)
        trainset_4 = torch.utils.data.Subset(trainset, odds4)
        trainset_5 = torch.utils.data.Subset(trainset, odds5)
        # trainset_2 = torch.utils.data.Subset(trainset, odds)

        trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=4,
                                                    shuffle=True, num_workers=2)
        trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=4,
                                                    shuffle=True, num_workers=2)
        trainloader_3 = torch.utils.data.DataLoader(trainset_3, batch_size=4,
                                                   shuffle=True, num_workers=2)
        trainloader_4 = torch.utils.data.DataLoader(trainset_4, batch_size=4,
                                                    shuffle=True, num_workers=2)
        trainloader_5 = torch.utils.data.DataLoader(trainset_5, batch_size=4,
                                                    shuffle=True, num_workers=2)
        

        time.sleep(1000)
        exit()

        sched = AFL.Scheduler(**configs[0])
        time.sleep(100)
        exit()
        outputs = AFL.Scheduler.run_multiple(configs, pool_size=2)
        

        # Replace class names with strings for serialization
        for i in outputs:
            i[1]['clients']['client'] = i[1]['clients']['client'].__name__
            i[1]['clients']['f_type'] = i[1]['clients']['f_type'].__name__
            i[1]['server'] = i[1]['server'].__name__

        # Write raw data to file
        outputs += completed_runs
        with open(data_file, 'w') as f:
            json.dump(outputs, f)

    # Load raw data from file
    outputs2 = ''
    with open(data_file, 'r') as f:
        outputs2 = json.load(f)

    # Process data into dataframe
    dfs = []
    for out in outputs2:
        name = out[1]['name']
        local_df = pd.DataFrame(out[0], columns=['round', 'accuracy', 'loss'])
        local_df['name'] = name.split('-')[-1]
        dfs.append(local_df)
    server_df = pd.concat(dfs, ignore_index=True)

    graph_file = graphs_path / f'{exp_name}.png'
    # Visualize data
    fig = plt.figure(figsize=(8, 6))
    g = sns.lineplot(data=server_df, x='round', y='accuracy', hue='name')
    plt.title('Different number of clients in async Learning. Cifar100 - ResNet9')
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    g.legend_.set_title(None)
    plt.savefig(graph_file)
    plt.close(fig)
