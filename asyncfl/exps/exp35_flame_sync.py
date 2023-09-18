import argparse
import copy
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
    parser.add_argument("-s", "--show-plots", help="Show generated plots",
                    action="store_true")
    parser.add_argument('-a', '--autocomplete',
                        help="Autocomplete missing experiments. Based on the results in the datafile, missing experiment will be run.", action='store_true')
    args = parser.parse_args()

    print('Exp 35: Flame synchronous version')

    (data_path := Path('.data')).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path('graphs')).mkdir(exist_ok=True, parents=True)
    exp_name = 'exp35_flame_sync'
    data_file = data_path / f'{exp_name}.json'

    if not args.o:
        # Define configuration

        pool_size = 4
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
        num_rounds = 300
        idx = 1
        repetitions = 1
        exp_id = 0
        # server_lr = 0.005
        server_lr = 0.01
        num_clients = 40

        attacks = [
            [AFL.NGClient, {'magnitude': 10,'sampler': 'uniform','sampler_args': {}}],
            # [AFL.RDCLient, {'a_atk':0.1, 'sampler': 'uniform', 'sampler_args': {}}],
        ]
        # print(num_clients //10)
        # exit(1)
        
        servers = [
            [AFL.FlameServer,{'learning_rate': server_lr, 'hist_size': int(num_clients*0.2)}],

        ]
        f0_keys = []

        # ct_clients = np.abs(np.random.normal(50, 20.0, num_clients - num_byz_nodes[0]))
        # f_clients = np.abs(np.random.normal(50, 20.0, num_byz_nodes[0]))
        # ct_clients = [1] * (num_clients - f)
        # print(ct_clients)

        # exit(1)
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
            #da{server[1]}'
            # if key_name not in f0_keys:
            #     f0_keys.append(key_name)
            exp_id += 1
            # configs.append({
            #     'exp_id': exp_id,
            #     'aggregation_type': 'sync',
            #     'client_participartion': 1,
            #     'name': f'{server_name}-sync-{key_name}',
            #     'num_rounds': num_rounds,
            #     'client_batch_size': -1,
            #     'eval_interval': 1,
            #     'clients': {
            #             'client': AFL.Client,
            #             'client_args': {
            #                 'learning_rate': server_lr,
            #                 'sampler': 'limitlabel',
            #                 'sampler_args': (7, 42)
            #             },
            #         'client_ct': ct_clients,
            #         'n': num_clients,
            #         'f': f,
            #         'f_type': atk[0],
            #         'f_args': atk[1],
            #         'f_ct': f_ct
            #     },
            #     'server': server[0],
            #     'server_args': server[1],
            #     'dataset_name': dataset,
            #     'model_name': model_name
            # })
            configs.append({
                'exp_id': exp_id,
                'aggregation_type': 'sync',
                'client_participartion': 0.2,
                'name': f'{server_name}-sync-{key_name}',
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

        
        # Run all experiments multithreaded
        completed_runs = []
        if args.autocomplete:
            print('Running mising experiments:')
            with open(data_file, 'r') as f:
                completed_runs = json.load(f)
                keys = [x[1]['exp_id'] for x in completed_runs]
                configs = [x for x in configs if x['exp_id'] not in keys]
                # @TODO: Append to output instead of overwriting

        outputs = AFL.Scheduler.run_multiple(configs, pool_size=pool_size, outfile=data_file, clear_file=not args.autocomplete, multi_thread=True)

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
    for out in outputs2:
        name = out[1]['name']
        interaction_events = out[0][3]
        ie_df = pd.DataFrame(interaction_events, columns=['client_id', 'wall_time', 'min_ct', 'client_ct'])
        ie_df['alg'] = name
        ie_df['round'] = ie_df.index
        interaction_dfs.append(ie_df)
        local_df = pd.DataFrame(
            out[0][0], columns=['round', 'accuracy', 'loss'])
        # local_df['name'] = f"{name.split('-')[-1]}"
        parts = name.split('-')[-1].split('_')
        print(name.split('-')[-2])
        print(parts)
        f = int(parts[0][1:])
        byz_type = 'None'
        if f:
            byz_type = parts[-1].upper()
        # print(name, parts, " :: ", byz_type)
        local_df['f'] = f
        local_df['byz_type'] = byz_type
        local_df['name'] = '-'.join([f'f{f}', byz_type])
        local_df_name = '-'.join(parts[-2:] + [name.split('-')[-2]])
        local_df['name'] = local_df_name
        print(f"name-> {'-'.join(parts[-2:])}")
        ie_df['name'] = local_df_name

        ct = [[x, name, 'clients'] for x in out[1]['clients']['client_ct']]
        ct += [[x, name, 'f_clients'] for x in out[1]['clients']['f_ct']]
        local_client_dist_df = pd.DataFrame(ct, columns=['ct', 'alg', 'type'])
        client_dist_dfs.append(local_client_dist_df)
        # local_bft_df = pd.DataFrame(out[0][2], columns=['action', 'client_id', 'lipschitz', 'round', 'is_byzantine', 'performance', 'global_score'])
        # local_bft_df = pd.DataFrame(out[0][2], columns=['server_age', 'client_id', 'grad_age', 'is_byzantine', 'client_weight_vec'])
        # local_bft_df['name'] = name
        # bft_dfs.append(local_bft_df)
        dfs.append(local_df)



    server_df = pd.concat(dfs, ignore_index=True)
    client_dist_df = pd.concat(client_dist_dfs, ignore_index=True)
    interaction_events_df = pd.concat(interaction_dfs, ignore_index=True)
    # bft_stats_df = pd.concat(bft_dfs, ignore_index=True)

    # bft_data = bft_stats_df.values
    # client_ids = bft_stats_df['client_id'].values
    # client_weight_vectors = bft_stats_df['client_weight_vec'].values
    # is_byz_vec= bft_stats_df['is_byzantine'].values

    # filename = 'mnist_byz_weights_capture.json'
    # # bft_stats_df.to_json(filename)
    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5}) # type: ignore
    # for f_num, byz_type in itertools.product(server_df['f'].unique(), server_df['byz_type'].unique()):
    for f_num in server_df['f'].unique():
        # local_df = server_df[(server_df['f']==f_num) & (server_df['byz_type'] == byz_type)]
        local_df = server_df[(server_df['f']==f_num)]

        graph_file = graphs_path / f'{exp_name}_{f_num}.png'
        print(f'Generating plot: {graph_file}')
        # Visualize data
        fig = plt.figure(figsize=(8, 6))
        g = sns.lineplot(data=local_df, x='round', y='accuracy', hue='name',errorbar=('ci', 95))
        plt.title(f'Algorithms f={f_num}')
        plt.xlabel('Rounds')
        plt.ylabel('Test accuracy')
        if g.legend_:
            g.legend_.set_title(None)
        plt.savefig(graph_file)
        if show_plots:
            plt.show()
        plt.close(fig)

        # plt.figure()
        # sns.kdeplot(data=client_dist_df, x='ct', hue='alg', fill=True, alpha=.5)

        # plt.title('ct distribution')
        # plt.show()

        single_interaction_df = interaction_events_df
        # single_interaction_df = interaction_events_df[interaction_events_df['alg'] == list(interaction_events_df['alg'].unique())[0]]
        print(single_interaction_df)

        plt.figure()
        graph_file = graphs_path / f'{exp_name}_client_kde.png'

        sns.kdeplot(data=single_interaction_df, x='client_ct')
        plt.title('Compute kde')
        plt.savefig(graph_file)
        plt.show()


        # Time based statistics
        print(single_interaction_df)
        print(local_df)

        combined_df = local_df.merge(single_interaction_df, how='left', left_on=['name','round'], right_on = ['name','round'])
        local_df['wall_time'] = single_interaction_df['wall_time']
        print(combined_df['name'].unique())

        graph_file = graphs_path / f'{exp_name}_wall_time.png'
        fig = plt.figure(figsize=(8, 6))
        g = sns.lineplot(data=combined_df, x='wall_time', y='accuracy', hue='name',errorbar=('ci', 95))
        plt.title(f'Algorithms f={f_num}')
        plt.xlabel('Time (s)')
        plt.ylabel('Test accuracy')
        if g.legend_:
            g.legend_.set_title(None)
        plt.savefig(graph_file)
        plt.show()
        plt.close(fig)
 

    # print(bft_stats_df.head(1))
    # fig = plt.figure(figsize=(8, 6))
    # g = sns.scatterplot(data=bft_stats_df, x='server_age', y='client_id', style='is_byzantine')
    # plt.title('Telerig')
    # plt.xlabel('Rounds')
    # plt.ylabel('Test accuracy')
    # # g.legend_.set_title(None)
    # if show_plots:
    #     plt.show()
    # plt.close(fig)




    # graph_file = graphs_path / f'{exp_name}_byz_perf.png'
    # print(f'Generating plot: {graph_file}')
    # sns.set_theme(style="white", palette="Dark2", font_scale=1, rc={"lines.linewidth": 2.5}) # type: ignore
    # g = sns.relplot(data=bft_stats_df, x="round", y="performance", height=2, aspect=6, hue="action", row='name', style='is_byzantine')
    # axes = g.axes.flatten()
    # for ax in axes:
    #     ax.axhline(0.0, ls='--', linewidth=1, color='red')
    # plt.savefig(graph_file)
    # if show_plots:
    #     plt.show()
    # plt.close(fig)

    # graph_file = graphs_path / f'{exp_name}_byz_lips.png'
    # print(f'Generating plot: {graph_file}')
    # sns.set_theme(style="white", palette="Dark2", font_scale=1, rc={"lines.linewidth": 2.5}) # type: ignore
    # g = sns.relplot(data=bft_stats_df, x="round", y="lipschitz", height=2, aspect=6, hue="action", row='name', style='is_byzantine')
    # plt.yscale('log')
    # # axes = g.axes.flatten()
    # # for ax in axes:
    # #     ax.axhline(0.0, ls='--', linewidth=1, color='red')
    # plt.savefig(graph_file)
    # if show_plots:
    #     plt.show()
    # plt.close(fig)

    # graph_file = graphs_path / f'{exp_name}_byz_global_score.png'
    # print(f'Generating plot: {graph_file}')
    # sns.set_theme(style="white", palette="Dark2", font_scale=1, rc={"lines.linewidth": 2.5}) # type: ignore
    # g = sns.relplot(data=bft_stats_df, x="round", y="global_score", height=2, aspect=6, hue="action", row='name', style='is_byzantine')
    # plt.ylim(0,20)
    # # plt.yscale('log')
    # # axes = g.axes.flatten()
    # # for ax in axes:
    # #     ax.axhline(0.0, ls='--', linewidth=1, color='red')
    # plt.savefig(graph_file)
    # if show_plots:
    #     plt.show()
    # plt.close(fig)

