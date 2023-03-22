import json
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "-omit", help="Omit running experiment",
                    action="store_true")
    args = parser.parse_args()

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp00_base_check_epoch_lr'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        # Run experiment
        # Define configurations
        configs = []
        # n = 3  # number of total clients
        f = 0  # number of byzantine clients
        num_rounds = 50*10
        idx = 1
        # Config for Cifar10 dataset
        repetitions = 3
        limit = 10
        for _r in range(repetitions):
            # for model_name in ['cifar100-resnet','cifar100-vgg','cifar100-lenet']:
            for n in [25,10, 5, 1]:
                for model_name in ['cifar100-resnet9']:
                # for model_name in ['cifar100-vgg']:

                    configs.append({
                            'name': f'afl-{model_name}-cifar100-n{n}',
                            'num_rounds': num_rounds,
                            'clients': {
                                'client': AFL.Client,
                                'client_args': {
                                    'sampler': 'uniform',
                                    'sampler_args':{                    
                                    }
                                },
                                'client_ct': [1] * (n - f),
                                'n': n,
                                'f': f,
                                'f_type': AFL.NGClient,
                                'f_args': {'magnitude': 10},
                                'f_ct': [1] * f
                            },
                            'server': AFL.Server,
                            'server_args': {
                            },
                            'dataset_name': 'cifar100',
                            'model_name': model_name
                        })

        # Run all experiments multithreaded
        outputs = AFL.Scheduler.run_multiple(configs, pool_size= 1)

        # Replace class names with strings for serialization
        for i in outputs:
            i[1]['clients']['client'] = i[1]['clients']['client'].__name__
            i[1]['clients']['f_type'] = i[1]['clients']['f_type'].__name__
            i[1]['server'] = i[1]['server'].__name__

        # Write raw data to file
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
        local_df['name'] = name
        dfs.append(local_df)
    server_df = pd.concat(dfs, ignore_index=True)

    graph_file = graphs_path / f'{exp_name}.png'
    # Visualize data
    plt.figure()
    sns.lineplot(data=server_df, x='round', y='accuracy', hue='name')
    plt.savefig(graph_file)
    plt.show()
