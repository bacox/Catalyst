import argparse
from pathlib import Path
import numpy as np
import json
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
# Turn interactive plotting off
plt.ioff()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "-omit", help="Omit running experiment",
                        action="store_true")
    parser.add_argument('-a', '--autocomplete',
                        help="Autocomplete missing experiments. Based on the results in the datafile, missing experiment will be run.", action='store_true')
    args = parser.parse_args()

    print('Exp 16: Run related work in Cifar10 dataset. Attacks: Negative gradient (magnitude=10) '
          +'and random perbutation (alpha_atk = 0.2). Learning rate follows from learning rate sweep: exp 14.'
          +'SaSGD, BAGSD, Kardam')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp16_c10_related_work'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        # Define configurations

        pool_size = 1

        configs = []
        model_name = 'cifar10-resnet9'
        dataset = 'cifar10'
        num_byz_nodes = [0, 5, 10]
        num_rounds = 2000
        idx = 1
        repetitions = 2
        exp_id = 0
        server_lr = 0.05
        num_clients = 50
        attacks = [
            [AFL.NGClient, {'magnitude': 10,'sampler': 'uniform','sampler_args': {}}],
            [AFL.RDCLient, {'a_atk':0.2, 'sampler': 'uniform', 'sampler_args': {}}],
        ]
        servers = [
            [AFL.SaSGD,{'learning_rate': server_lr}],
            [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.01,}],
            [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': 15, 'aggr_mode': 'median'}]
        ]
        f0_keys = []

        for _r, f, server, atk in itertools.product(range(repetitions), num_byz_nodes, servers, attacks):
            # print(_r, f, n, s_lr, model_name)
            server_name = server[0].__name__
            attack_name = atk[0].__name__
            # print(server_name, attack_name)
            key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}'
            if f > 0:
                exp_id += 1
                configs.append({
                    'exp_id': exp_id,
                    'aggregation_type': 'async',
                    'name': f'{server_name}-{key_name}_{attack_name}',
                    'num_rounds': num_rounds,
                    'clients': {
                            'client': AFL.Client,
                            'client_args': {
                                'learning_rate': server_lr,
                                'sampler': 'uniform',
                                'sampler_args': {
                                }
                            },
                        'client_ct': [1] * (num_clients - f),
                        'n': num_clients,
                        'f': f,
                        'f_type': atk[0],
                        'f_args': atk[1],
                        'f_ct': [1] * f
                    },
                    'server': server[0],
                    'server_args': server[1],
                    'dataset_name': dataset,
                    'model_name': model_name
                })
                
            f = 0
            key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}'
            if key_name not in f0_keys:
                f0_keys.append(key_name)
                exp_id += 1
                configs.append({
                    'exp_id': exp_id,
                    'aggregation_type': 'async',
                    'name': f'{server_name}-{key_name}',
                    'num_rounds': num_rounds,
                    'clients': {
                            'client': AFL.Client,
                            'client_args': {
                                'learning_rate': server_lr,
                                'sampler': 'uniform',
                                'sampler_args': {
                                }
                            },
                        'client_ct': [1] * (num_clients - f),
                        'n': num_clients,
                        'f': f,
                        'f_type': AFL.RDCLient,
                        'f_args': {'a_atk':0.2,
                        'sampler': 'uniform',
                                'sampler_args': {
                                }
                        },
                        'f_ct': [1] * f
                    },
                    'server': server[0],
                    'server_args': server[1],
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

        outputs = AFL.Scheduler.run_multiple(configs, pool_size=pool_size)

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
        local_df = pd.DataFrame(
            out[0][0], columns=['round', 'accuracy', 'loss'])
        # local_df['name'] = f"{name.split('-')[-1]}"
        parts = name.split('-')[-1].split('_')
        f = int(parts[0][1:])
        byz_type = 'None'
        if f:
            byz_type = parts[-1].upper()
        # print(name, parts, " :: ", byz_type)
        local_df['f'] = f
        local_df['byz_type'] = byz_type
        local_df['name'] = '-'.join([f'f{f}', byz_type])
        if f:
            dfs.append(local_df)
        else:
            for f in [1,2,5,10]:
                local_df_update = local_df.copy()
                local_df_update['f'] = f
                dfs.append(local_df_update)
            # local_df = local_df.copy()
            # local_df['']
    server_df = pd.concat(dfs, ignore_index=True)
    server_df = server_df[server_df['f'].isin([1,2, 5,10])]

    dfs_server_age = []
    for out in outputs2:
        name = out[1]['name']
        local_df = pd.DataFrame(
            out[0][1], columns=['round', 'client', 'model_age'])
        local_df['name'] = f"{name.split('-')[-1]}"
        dfs_server_age.append(local_df)
    model_age_df = pd.concat(dfs_server_age, ignore_index=True)
    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5}) # type: ignore
    graph_file = graphs_path / f'{exp_name}.png'
    print(f'Generating plot: {graph_file}')
    # Visualize data
    fig = plt.figure(figsize=(8, 6))
    g = sns.lineplot(data=server_df, x='round', y='accuracy', hue='name',errorbar=('ci', 25))
    plt.title('Effect of Byzantine Nodes')
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    g.legend_.set_title(None)
    plt.savefig(graph_file)
    plt.close(fig)

    graph_file = graphs_path / f'{exp_name}_splitted.png'
    print(f'Generating plot: {graph_file}')
    # Visualize data
    fig = plt.figure(figsize=(8, 6))
    g = sns.relplot(data=server_df, x='round', y='accuracy', hue='byz_type', col='f', kind='line', errorbar=('ci', 25))
    g.set_axis_labels("Rounds", "Accuracy").set_titles("F = {col_name}").tight_layout(w_pad=0)
    g.legend.set_title('Attack')
    g.fig.subplots_adjust(top=0.8) # adjust the Figure in g
    g.fig.suptitle(f'Effects of byzantine nodes. MNIST, 50 nodes, lr=0.1')
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    plt.savefig(graph_file)
    plt.close(fig)

    # graph_file = graphs_path / f'{exp_name}_model_age.png'
    # print(f'Generating plot: {graph_file}')
    # # Visualize data
    # fig = plt.figure(figsize=(8, 6))
    # g = sns.lineplot(data=model_age_df, x='round', y='model_age', hue='name')
    # plt.title('Model Age')
    # plt.xlabel('Rounds')
    # plt.ylabel('Model age')
    # g.legend_.set_title(None)
    # plt.savefig(graph_file)
    # plt.close(fig)

    # graph_file = graphs_path / f'{exp_name}_model_age_hist.png'
    # print(f'Generating plot: {graph_file}')
    # # Visualize data
    # fig = plt.figure(figsize=(8, 6))
    # # g = sns.lineplot(data=model_age_df, x='round', y='model_age', hue='name')
    # g = sns.histplot(data=model_age_df, x="model_age", kde=True, hue='name')
    # plt.title('Model Age')
    # plt.xlabel('Model Age')
    # plt.ylabel('Density')
    # # g.legend_.set_title(None)
    # plt.savefig(graph_file)
    # plt.close(fig)

    # graph_file = graphs_path / f'{exp_name}_client_hist.png'
    # print(f'Generating plot: {graph_file}')
    # # Visualize data
    # fig = plt.figure(figsize=(8, 6))
    # # g = sns.lineplot(data=model_age_df, x='round', y='model_age', hue='name')
    # g = sns.histplot(data=model_age_df, x="client", kde=True, hue='name')
    # plt.title('Client contributions')
    # plt.xlabel('Client contributions')
    # plt.ylabel('Density')
    # # g.legend_.set_title(None)
    # plt.savefig(graph_file)
    # plt.close(fig)
