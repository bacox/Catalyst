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

    print('Exp 21: Test the bft stats using the mnist dataset')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp21_mnist_bft_stats'
    data_file = data_path / f'{exp_name}.json'

    # args.o = True

    if not args.o:
        # Define configurations

        pool_size = 2

        configs = []
        model_name = 'mnist-cnn'
        dataset = 'mnist'
        num_byz_nodes = [1]
        # num_byz_nodes = [0]
        num_rounds = 25
        idx = 1
        repetitions = 1
        exp_id = 0
        server_lr = 0.05
        num_clients = 10
        attacks = [
            # [AFL.NGClient, {'magnitude': 10,'sampler': 'uniform','sampler_args': {}}],
            [AFL.RDCLient, {'a_atk':0.2, 'sampler': 'uniform', 'sampler_args': {}}],
        ]
        
        servers = [
            [AFL.Telerig,{'learning_rate': server_lr, 'damp_alpha': 0.3, 'eps': 0.05}],
            [AFL.Telerig,{'learning_rate': server_lr, 'damp_alpha': 0.3, 'eps': 0.5}],
            [AFL.Telerig,{'learning_rate': server_lr, 'damp_alpha': 0.3, 'eps': 1.0}],
            [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.01,}],

            # [AFL.Telerig,{'learning_rate': server_lr, 'damp_alpha': 0.5,}],
        ]
        f0_keys = []

        for _r, f, server, atk in itertools.product(range(repetitions), num_byz_nodes, servers, attacks):
            # print(_r, f, n, s_lr, model_name)
            server_name = server[0].__name__
            attack_name = atk[0].__name__
            if server_name != 'Kardam':
                key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}_da{server[1]["damp_alpha"]}_eps{server[1]["eps"]}'
            else:
                key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}_da{server[1]["damp_alpha"]}_eps0'
            # if key_name not in f0_keys:
            #     f0_keys.append(key_name)
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

        outputs = AFL.Scheduler.run_multiple(configs, pool_size=pool_size, outfile=data_file, clear_file=not args.autocomplete)

    # Load raw data from file
    outputs2 = ''
    with open(data_file, 'r') as f:
        outputs2 = json.load(f)

    if not args.o:
        exit()
    # Process data into dataframe
    dfs = []
    bft_dfs = []
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
        local_df['name'] = parts[-1]
        local_bft_df = pd.DataFrame(out[0][2], columns=['action', 'client_id', 'lipschitz', 'round', 'is_byzantine', 'performance', 'global_score'])
        local_bft_df['name'] = name
        bft_dfs.append(local_bft_df)
        dfs.append(local_df)


    server_df = pd.concat(dfs, ignore_index=True)
    bft_stats_df = pd.concat(bft_dfs, ignore_index=True)

    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5}) # type: ignore
    graph_file = graphs_path / f'{exp_name}.png'
    print(f'Generating plot: {graph_file}')
    # Visualize data
    fig = plt.figure(figsize=(8, 6))
    g = sns.lineplot(data=server_df, x='round', y='accuracy', hue='name',errorbar=('ci', 95))
    plt.title('Telerig')
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    g.legend_.set_title(None)
    plt.savefig(graph_file)
    plt.close(fig)


    graph_file = graphs_path / f'{exp_name}_byz_perf.png'
    print(f'Generating plot: {graph_file}')
    sns.set_theme(style="white", palette="Dark2", font_scale=1, rc={"lines.linewidth": 2.5}) # type: ignore
    g = sns.relplot(data=bft_stats_df, x="round", y="performance", height=2, aspect=6, hue="action", row='name', style='is_byzantine')
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0.0, ls='--', linewidth=1, color='red')
    plt.savefig(graph_file)
    plt.close(fig)

    graph_file = graphs_path / f'{exp_name}_byz_lips.png'
    print(f'Generating plot: {graph_file}')
    sns.set_theme(style="white", palette="Dark2", font_scale=1, rc={"lines.linewidth": 2.5}) # type: ignore
    g = sns.relplot(data=bft_stats_df, x="round", y="lipschitz", height=2, aspect=6, hue="action", row='name', style='is_byzantine')
    plt.yscale('log')
    # axes = g.axes.flatten()
    # for ax in axes:
    #     ax.axhline(0.0, ls='--', linewidth=1, color='red')
    plt.savefig(graph_file)
    plt.close(fig)

    graph_file = graphs_path / f'{exp_name}_byz_global_score.png'
    print(f'Generating plot: {graph_file}')
    sns.set_theme(style="white", palette="Dark2", font_scale=1, rc={"lines.linewidth": 2.5}) # type: ignore
    g = sns.relplot(data=bft_stats_df, x="round", y="global_score", height=2, aspect=6, hue="action", row='name', style='is_byzantine')
    plt.yscale('log')
    # axes = g.axes.flatten()
    # for ax in axes:
    #     ax.axhline(0.0, ls='--', linewidth=1, color='red')
    plt.savefig(graph_file)
    plt.close(fig)

