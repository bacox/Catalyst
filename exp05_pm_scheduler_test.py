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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "-omit", help="Omit running experiment",
                        action="store_true")
    args = parser.parse_args()

    print('Exp 05: Testing the PoolManager scheduler.')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp05_pm_sched'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        # Define configurations
        configs = []
        model_list = ['mnist-cnn']
        dataset = 'mnist'
        f = 0  # number of byzantine clients
        # num_rounds = 50*10
        num_rounds = 10
        idx = 1
        repetitions = 3
        limit = 10
        num_clients = [5]
        # num_clients = [50, 25, 10, 5, 1]
        for _r in range(repetitions):
            for n in num_clients:
                for model_name in model_list:
                    configs.append({
                        'task_cap': 1.0,
                        'aggregation_type': 'sync',
                        'client_participartion': 0.2,
                        'name': f'afl-{model_name}-{dataset}-n{n}_sync',
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
                        'dataset_name': dataset,
                        'model_name': model_name
                    })
                    configs.append({
                        'task_cap': 0.5,
                        'aggregation_type': 'async',
                        'client_participartion': 0.2,
                        'name': f'afl-{model_name}-{dataset}-n{n}_async',
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
                        'dataset_name': dataset,
                        'model_name': model_name
                    })
                    configs.append({
                        'task_cap': 0.5,
                        'aggregation_type': 'async',
                        'client_participartion': 0.2,
                        'name': f'afl-{model_name}-{dataset}-n{n}_async',
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
                        'dataset_name': dataset,
                        'model_name': model_name
                    })

        # Run all experiments multithreaded
        outputs = AFL.Scheduler.run_pm(configs, pool_size=4)

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
