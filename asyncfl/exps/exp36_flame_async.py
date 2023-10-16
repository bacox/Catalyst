import copy
from pathlib import Path
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import numpy as np
import json
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools

from asyncfl.util import cli_options, dict_hash
# Turn interactive plotting off
plt.ioff()

if __name__ == '__main__':
    args = cli_options()

    print('Exp 36: Flame asynchronous version')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp36_flame_async'
    data_file = data_path / f'{exp_name}.json'



    # Dev notes:
    # @TODO: generalize config generation
    # What do we need for a config?
    # Description of servers
    # Description of clients
    # Description of attackers
    # General description of scenario

    # Then:
    # - Generate client_ct
    # - Generate configs
    # - Execute configs

    if not args.o:
        # Define configuration
        # Single threaded is suggested when running with 100 clients
        multi_thread= True
        pool_size = 5
        configs = []
        # model_name = 'cifar10-resnet9'
        # model_name = 'cifar10-resnet18'
        # model_name = 'cifar10-lenet'
        # dataset = 'cifar10'
        model_name = 'mnist-cnn'
        dataset = 'mnist'
        # num_byz_nodes = [0, 1, 3]
        num_byz_nodes = [1]
        # num_byz_nodes = [0]
        num_rounds = 200
        idx = 1
        repetitions = 3
        exp_id = 0
        # server_lr = 0.005
        server_lr = 0.1
        # num_clients = 50
        num_clients = 10

        attacks = [
            [AFL.NGClient, {'magnitude': 10,'sampler': 'uniform','sampler_args': {}}],
            # [AFL.RDCLient, {'a_atk':0.1, 'sampler': 'uniform', 'sampler_args': {}}],
        ]

        servers = [
            [AFL.Kardam,{'learning_rate': 0.1, 'damp_alpha': 0.1,}],
            [AFL.Kardam,{'learning_rate': 0.1, 'damp_alpha': 0.5,}],
            # [AFL.Kardam,{'learning_rate': 0.01, 'damp_alpha': 0.1,}],
            # [AFL.Kardam,{'learning_rate': 0.005, 'damp_alpha': 0.1,}],
            # [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.02,}],
            # [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.05,}],
            # [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.1,}],
            # [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.2,}],
            # [AFL.Kardam,{'learning_rate': server_lr, 'damp_alpha': 0.5,}],
            # # [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2), 'min_cluster_size': 2}],
            # # [AFL.FlameServer,{'learning_rate': 0.1, 'hist_size': 11, 'min_cluster_size': 11}],
            [AFL.FlameServer,{'learning_rate': 0.1, 'hist_size': 11, 'min_cluster_size': 11}],
            [AFL.FlameServer,{'learning_rate': 0.05, 'hist_size': 11, 'min_cluster_size': 11}],
            [AFL.FlameServer,{'learning_rate': 0.025, 'hist_size': 11, 'min_cluster_size': 11}],
            # # [AFL.FlameServer,{'learning_rate': 0.01, 'hist_size': 11, 'min_cluster_size': 11}],
            # # [AFL.FlameServer,{'learning_rate': 0.005, 'hist_size': 11, 'min_cluster_size': 11}],
            # # [AFL.FlameServer,{'learning_rate': 0.001, 'hist_size': 11, 'min_cluster_size': 11}],
            # # [AFL.FlameServer,{'learning_rate': 0.2, 'hist_size': 4, 'min_cluster_size': 3}],
            # # [AFL.FlameServer,{'learning_rate': 0.3, 'hist_size': 4, 'min_cluster_size': 3}],
            # # [AFL.FlameServer,{'learning_rate': 0.4, 'hist_size': 4, 'min_cluster_size': 3}],
            # # [AFL.FlameServer,{'learning_rate': 0.5, 'hist_size': 4, 'min_cluster_size': 3}],
            # # [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2), 'min_cluster_size': 4}],
            # # [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2), 'min_cluster_size': 5}],
            # # [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2), 'min_cluster_size': 6}],
            # # [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2), 'min_cluster_size': 7}],
            # # [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2), 'min_cluster_size': 8}],

            # [AFL.SaSGD,{'learning_rate': server_lr}],
            # [AFL.Server,{'learning_rate': server_lr}],
            # [AFL.FedAsync,{'learning_rate': 0.5},],
            # [AFL.FedAsync,{'learning_rate': 0.1},],
            # [AFL.FedAsync,{'learning_rate': 0.05},],
            # [AFL.FedAsync,{'learning_rate': 0.01},],
            # # [AFL.FedWait,{'learning_rate': server_lr}],
            # # [AFL.Server,{'learning_rate': server_lr}],
            # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': num_clients // 3}],
            # # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': num_clients // 3, 'aggr_mode': 'trmean'}],
            # # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': num_clients // 8}],
            # # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': num_clients // 16}],

        ]
        f0_keys = []


        generated_ct = {}

        # @TODO: Make sure they have exactly the same schedule!!
        # 1. Generate the same client schedule
        # 2. Generate the same data distribution
        #
        # Q? Can this be fixed internally by having the same seed?

        for _r, f, server, atk in itertools.product(range(repetitions), num_byz_nodes, servers, attacks):
            ct_key = f'{num_clients}-{f}'
            if ct_key not in generated_ct.keys():
                ct_clients = np.abs(np.random.normal(100, 40.0, num_clients - f))
                f_ct = np.abs(np.random.normal(100, 40.0, f))
                generated_ct[ct_key] = [ct_clients, f_ct]
            ct_clients, f_ct = copy.deepcopy(generated_ct[ct_key])

            # Round robin
            # f_ct = [1] * f
            # ct_clients = [1] * (num_clients - f)

            server_name = server[0].__name__
            attack_name = atk[0].__name__
            key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}_{server_name}'
            if server[0] == AFL.FlameServer:
                key_name += f'_{server[1]["hist_size"]}'
            else:
                key_name += f'_0'
            exp_id += 1
 
            configs.append({
                'exp_id': exp_id,
                'aggregation_type': 'async',
                'client_participartion': 0.2,
                'name': f'{server_name}-async-{key_name}',
                'num_rounds': num_rounds,
                'client_batch_size': -1,
                'eval_interval': 5,
                'clients': {
                        'client': AFL.Client,
                        'client_args': {
                            'learning_rate': server_lr,
                            'sampler': 'limitlabel',
                            'sampler_args': (7, 42)
                        },
                    'client_ct': ct_clients,
                    'n': num_clients,
                    'f': f,
                    'f_type': atk[0],
                    'f_args': atk[1],
                    'f_ct': f_ct
                },
                'server': server[0],
                'server_args': server[1],
                'dataset_name': dataset,
                'model_name': model_name
            })


        # for cfg in configs:
        #     print(dict_hash(cfg, exclude_keys=['exp_id']))

        # exit(0)
        # Run all experiments multithreaded
        outputs = AFL.Scheduler.run_multiple(configs, pool_size=pool_size, outfile=data_file, clear_file=not args.autocomplete, multi_thread=multi_thread, autocomplete=args.autocomplete)

    # Load raw data from file
    outputs2 = ''
    with open(data_file, 'r') as f:
        outputs2 = json.load(f)

    if not args.o:
        exit()
    
    show_plots = False
    if args.show_plots:
        show_plots = True


    # Process data into dataframe
    dfs = []
    bft_dfs = []
    client_dist_dfs = []
    interaction_dfs = []
    for running_stats, cfg_data in outputs2:
        name = cfg_data['name']
        min_cluster_size = 0
        if 'min_cluster_size' in cfg_data['server_args']:
            min_cluster_size = int(cfg_data['server_args']['min_cluster_size'])
        interaction_events = running_stats[3]
        ie_df = pd.DataFrame(interaction_events, columns=['client_id', 'wall_time', 'min_ct', 'client_ct'])
        ie_df['alg'] = name
        ie_df['round'] = ie_df.index
        interaction_dfs.append(ie_df)
        local_df = pd.DataFrame(
            running_stats[0], columns=['round', 'accuracy', 'loss'])
        parts = name.split('-')[-1].split('_')

        # print(parts)
        # pp.pprint(cfg_data)

        server_lr = cfg_data['server_args']['learning_rate']
        num_rounds = cfg_data['num_rounds']
        kardam_damp = ''
        if 'damp_alpha' in cfg_data['server_args']:
            kardam_damp = cfg_data['server_args']['damp_alpha']
        f = int(parts[0][1:])
        byz_type = 'None'
        if f:
            byz_type = parts[-1].upper()
        local_df['f'] = f
        local_df['byz_type'] = byz_type
        local_df_name = f'{parts[-2]}-f{f}-cs{min_cluster_size}-{server_lr}-{kardam_damp}'
        print(local_df_name, parts)
        local_df['name'] = local_df_name
        ie_df['name'] = local_df_name

        ct = [[x, name, 'clients'] for x in cfg_data['clients']['client_ct']]
        ct += [[x, name, 'f_clients'] for x in cfg_data['clients']['f_ct']]
        local_client_dist_df = pd.DataFrame(ct, columns=['ct', 'alg', 'type'])
        client_dist_dfs.append(local_client_dist_df)

        dfs.append(local_df)



    server_df = pd.concat(dfs, ignore_index=True)
    client_dist_df = pd.concat(client_dist_dfs, ignore_index=True)
    interaction_events_df = pd.concat(interaction_dfs, ignore_index=True)

    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5}) # type: ignore
    fig_size = (8*2, 6*2)
    local_df = server_df

    graph_file = graphs_path / f'{exp_name}_byz.png'
    print(f'Generating plot: {graph_file}')
    # Visualize data
    fig = plt.figure(figsize=fig_size)
    g = sns.lineplot(data=local_df, x='round', y='accuracy', hue='name',errorbar=('ci', 95))
    plt.title(f'Algorithms Flameserver')
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    if g.legend_:
        g.legend_.set_title(None)
    plt.savefig(graph_file)
    print(f'Saving figure at {graph_file}')

    if show_plots:
        plt.show()
    plt.close(fig)


    single_interaction_df = interaction_events_df
    # print(single_interaction_df)

    plt.figure()
    graph_file = graphs_path / f'{exp_name}_client_kde.png'

    sns.kdeplot(data=single_interaction_df, x='client_ct')
    plt.title('Compute kde')
    plt.savefig(graph_file)
    print(f'Saving figure at {graph_file}')

    plt.show()


    # Time based statistics
    # print(single_interaction_df)
    # print(local_df)

    combined_df = local_df.merge(single_interaction_df, how='left', left_on=['name','round'], right_on = ['name','round'])
    local_df['wall_time'] = single_interaction_df['wall_time']
    print(combined_df['name'].unique())

    graph_file = graphs_path / f'{exp_name}_wall_time.png'
    fig = plt.figure(figsize=fig_size)
    g = sns.lineplot(data=combined_df, x='wall_time', y='accuracy', hue='name',errorbar=('ci', 95))
    plt.title(f'Algorithms Flameserver')
    plt.xlabel('Time (s)')
    plt.ylabel('Test accuracy')
    if g.legend_:
        g.legend_.set_title(None)
    plt.savefig(graph_file)
    print(f'Saving figure at {graph_file}')
    plt.show()
    plt.close(fig)
 