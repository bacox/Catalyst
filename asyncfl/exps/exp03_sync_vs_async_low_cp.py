import json
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()
import numpy as np
from pathlib import Path
import argparse
import time
# import torch
# torch.multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "-omit", help="Omit running experiment",
                        action="store_true")
    args = parser.parse_args()

    print('Exp 3: Async training with 20% client participation')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp03_low_client_participation'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        start_time = time.time()
        outputs = []
        # Define configurations
        configs = []

        # Shared configs:
        f = 0  # number of byzantine clients
        num_rounds = 1000
        idx = 1
        repetitions = 2
        num_clients = [20, 10, 5]
        for _r in range(repetitions):
            for n in num_clients:
                for model_name in ['cifar100-resnet9']:
                    configs.append({
                        'aggregation_type': 'asynchronous',
                        'name': f'afl-{model_name}-cifar100-n{n}_async',
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
                        'server': AFL.Server,
                        'server_args': {
                        },
                        'dataset_name': 'cifar100',
                        'model_name': model_name
                    })

        # Run all experiments multithreaded
        # outputs = AFL.Scheduler.run_sync(configs)
        outputs += AFL.Scheduler.run_multiple(configs, pool_size=2)

        # Define configurations
        configs = []
        for _r in range(repetitions):
            # for n in [50, 25, 10, 2]:
            # for n in [10, 5, 2, 1]:
            for n in num_clients:
                for model_name in ['cifar100-resnet9']:
                    configs.append({
                        'client_participartion': 0.2,
                        'aggregation_type': 'synchronous',
                        'name': f'afl-{model_name}-cifar100-n{n}_sync',
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
                        'server': AFL.Server,
                        'server_args': {
                        },
                        'dataset_name': 'cifar100',
                        'model_name': model_name
                    })

        # Run all experiments multithreaded
        outputs += AFL.Scheduler.run_sync(configs)

        

        # Replace class names with strings for serialization
        for i in outputs:
            i[1]['clients']['client'] = i[1]['clients']['client'].__name__
            i[1]['clients']['f_type'] = i[1]['clients']['f_type'].__name__
            i[1]['server'] = i[1]['server'].__name__
            # i[1]['aggregation_type'] = 'synchronous'

        # Write raw data to file
        with open(data_file, 'w') as f:
            json.dump(outputs, f)
        
        print(f"--- Running time of experiment: {(time.time() - start_time):.2f} seconds ---")


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
    plt.title('Synchronous Learning vs Async Learning. Cifar100 - ResNet9')
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    g.legend_.set_title(None)
    plt.savefig(graph_file)
    plt.close(fig)
    # plt.show()
