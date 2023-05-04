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
    exp_name = 'exp01_async_effect'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        # Run experiment
        # Define configurations
        configs = []
        n = 2  # number of total clients
        f = 0  # number of byzantine clients
        num_rounds = 100
        idx = 1
        # Config for Cifar10 dataset
        repetitions = 1
        limit = 10
        for _r in range(repetitions):
            # for damp_alpha in [0.1, 0.2, 0.025,0.5]:
            for damp_alpha in np.linspace(0.001, 0.10, num=10):
                limit = 10
                configs.append({
                    'name': f'afl-cifar100-n{n}-a{damp_alpha}',
                    'num_rounds': num_rounds,
                    'clients': {
                        'client': AFL.Client,
                        'client_args': {
                            'sampler': 'n_labels',
                            'sampler_args':{
                                'limit': limit
                            }
                        },
                        'client_ct': [1] * (n - f),
                        # 'client_ct': [1] * (n - f-1) + [0.1],
                        # 'client_ct': list(np.random.uniform(0.1, 10, n - f)),
                        # 'client_ct': list(np.abs(np.random.normal(50, 5, n - f))),
                        'n': n,
                        'f': f,
                        'f_type': AFL.NGClient,
                        'f_args': {'magnitude': 10},
                        'f_ct': [1] * f
                    },
                    'server': AFL.Kardam,
                    'server_args': {
                        'damp_alpha': damp_alpha
                    },
                    'dataset_name': 'cifar100',
                    'model_name': 'cifar100-lenet'
                })
            configs.append({
                    'name': f'afl-base-cifar100-n{n}',
                    'num_rounds': num_rounds,
                    'clients': {
                        'client': AFL.Client,
                        'client_args': {
                            'sampler': 'n_labels',
                            'sampler_args':{                    
                                'limit': limit
                            }
                        },
                        'client_ct': [1] * (n - f),
                        # 'client_ct': [1] * (n - f-1) + [0.1],
                        # 'client_ct': list(np.random.uniform(0.1, 10, n - f)),
                        # 'client_ct': list(np.abs(np.random.normal(50, 5, n - f))),
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
                    'model_name': 'cifar100-lenet'
                })

        # Run all experiments multithreaded
        outputs = AFL.Scheduler.run_multiple(configs, pool_size=8)

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
