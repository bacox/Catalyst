import copy
from pathlib import Path
from pprint import PrettyPrinter
from functools import partial

import wandb

from asyncfl.exps import get_exp_project_name

from tqdm import tqdm
from asyncfl.dataloader import load_mnist

# from asyncfl.pixel_client import LocalMaliciousUpdate

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



if __name__ == "__main__":
    args = cli_options()

    print("Exp 61 - Correct sweep")

    exp_name = "exp61_correct_sweep_v2"

    (data_path := Path(".data")).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path("graphs") / exp_name).mkdir(exist_ok=True, parents=True)
    data_file = data_path / f"{exp_name}.json"

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
        multi_thread = True
        pool_size = 5
        configs = []
        dist_check = False
        dist_check_exit = False
        # model_name = 'cifar10-resnet9'
        # model_name = 'cifar10-resnet18'
        # model_name = 'cifar10-lenet'
        # dataset = 'cifar10'
        model_name = "mnist-cnn"
        dataset = "mnist"
        # num_byz_nodes = [0, 1, 3]
        # num_byz_nodes = [1]
        # num_byz_nodes = [0]
        num_rounds = 140
        idx = 1  # Most likely should not be changed in most cases
        repetitions = 3
        exp_id = 0
        # server_lr = 0.005
        server_lr = 1
        # num_clients = 50
        # num_clients = 10
        ff = 3

        reporting = True # Report to wandb

        # Sampling labels limit
        # limit = [2,3,4,5,6,7,8,9,10]
        limit = [2,4]
        var_sets = [
            # {"num_clients": 40, "num_byz_nodes": 1, "flame_hist": 3},
            # {"num_clients": 40, "num_byz_nodes": 4, "flame_hist": 3},
            {"num_clients": 40, "num_byz_nodes": 8, "flame_hist": 3},
            # {"num_clients": 100, "num_byz_nodes": 30, "flame_hist": 3},
            # {"num_clients": 30, "num_byz_nodes": 10, "flame_hist": 3},
            # {"num_clients": 10, "num_byz_nodes": 3, "flame_hist": 3},
        ]

        byz_uniform_sampler = {'sampler': 'uniform', 'sampler_args': {}}


        attacks = [
            [AFL.NGClient, {**{"magnitude": 10}, **byz_uniform_sampler}],
        ]

        servers = [
            [AFL.SemiAsync, {"learning_rate": server_lr, "k": 20, "disable_alpha": False, 'reporting': reporting}, 'semi-async'],
            [AFL.FlameNaiveBaseline,
                {
                    "learning_rate": server_lr, 
                    "k": 1, 
                    "disable_alpha": True,
                    "alg_version": 'B2',
                    'reporting': reporting
                },
                "semi-async"
            ]
        ]

        f0_keys = []
        generated_ct = {}

        # @TODO: BASGD alg works with gradients. In the implementation we use weights. This is a difference.

        for _r, server, var_set, atk, limit_val in itertools.product(range(repetitions), servers, var_sets, attacks, limit):



            num_clients, f, fh = var_set.values()
            ct_key = f"{num_clients}-{f}-{fh}-{_r}"
            # f = ff
            # print(n, f, fh)
            # ct_key = f'{num_clients}-{f}'
            if ct_key not in generated_ct.keys():

                # First create the byzantine client speeds
                f_ct = np.abs(np.random.normal(200, 5, f))

                # Create the benign client speeds
                b = num_clients - f

                f2 = min(2*f,b)
                ct_f2 = np.abs(np.random.normal(600, 5, f2))
                
                ct_rest = np.abs(np.random.normal(400, 5, max(b - f2, 0)))
                ct_clients = np.concatenate((ct_f2, ct_rest), axis=0)
                ct_clients = sorted(ct_clients)
                print(f'{len(ct_clients)} and {b=}')
                # print(sorted(ct_clients))
                # print(ct_clients)
                # exit()
                assert len(ct_clients) == b
                # ct_clients = np.abs(np.random.normal(1000, 5, num_clients - f))
                f_ct = np.abs(np.random.normal(200, 5, f))
                generated_ct[ct_key] = [ct_clients, f_ct]
            ct_clients, f_ct = copy.deepcopy(generated_ct[ct_key])

            server_name = server[0].__name__
            attack_name = atk[0].__name__
            key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}_{server_name}'
            if server[0] == AFL.FlameServer:
                key_name += f'_{server[1]["hist_size"]}'
            else:
                key_name += f"_0"
            exp_id += 1

            rounds = num_rounds
            if server[2] != "semi-async":
                rounds = num_rounds * num_clients
            configs.append(
                {
                    "project": get_exp_project_name(),
                    'exp_name': exp_name,
                    "exp_id": exp_id,
                    "aggregation_type": server[2],
                    "client_participartion": 0.2,
                    "name": f"{server_name}-async-{key_name}",
                    "num_rounds": rounds,
                    "client_batch_size": -1,
                    "eval_interval": 5,
                    "clients": {
                        "client": AFL.Client,
                        'client_args': {'sampler': 'nlabels', 'sampler_args': {'limit': limit_val}},
                        # "client_args": {"learning_rate": server_lr, "sampler": "uniform", "sampler_args": {}},
                        "client_ct": ct_clients,
                        "n": num_clients,
                        # "f": ff,
                        'f': f,
                        "f_type": atk[0],
                        "f_args": atk[1],
                        "f_ct": f_ct,
                    },
                    "server": server[0],
                    "server_args": server[1],
                    "dataset_name": dataset,
                    "model_name": model_name,
                    "meta_data": {
                        'non_iid_limit': limit_val,
                        'num_byz_nodes': f,
                        'replication': _r,
                        }
                }
            )

        if dist_check:
            # This breaks the code!
            # @TODO: Fix
            
            AFL.Scheduler.plot_data_distribution_by_time([configs[0]], use_cache=False, filter_byzantine=True)
            AFL.Scheduler.plot_data_distribution_by_time([configs[0]], use_cache=True, filter_byzantine=False, plot_name_prefix='data_dist_byz')



            answer = input("Check the data distribution plot. Continue?")
            if answer.upper() in ["Y", "YES"]:
                print('Registered "yes"')
            else:
                print('Registered "no"')
                print('Exiting')
                exit()
                # Do action you need
            # else if answer.upper() in ["N", "NO"]:
        
        if dist_check_exit:
            print(f'Number of configs={len(configs)}')
            AFL.Scheduler.plot_data_distribution_by_time([configs[0]], use_cache=False, filter_byzantine=True)
            AFL.Scheduler.plot_data_distribution_by_time([configs[0]], use_cache=True, filter_byzantine=False, plot_name_prefix='data_dist_byz')
            exit()

        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project=get_exp_project_name(),
            
        #     # track hyperparameters and run metadata
        #     config={
        #     "learning_rate": server_lr,
        #     "architecture": "CNN",
        #     "dataset": model_name,
        #     "epochs": rounds,
        #     }
        # )

        outputs = AFL.Scheduler.run_multiple(
            configs,
            pool_size=pool_size,
            outfile=data_file,
            clear_file=not args.autocomplete,
            multi_thread=multi_thread,
            autocomplete=args.autocomplete,
        )
    wandb.finish()

    if False:
        AFL.Scheduler.plot_data_distribution_by_time([configs[0]], use_cache=False, filter_byzantine=True)
        AFL.Scheduler.plot_data_distribution_by_time([configs[0]], use_cache=True, filter_byzantine=False, plot_name_prefix='data_dist_byz')
    # Load raw data from file
    outputs2 = ""
    with open(data_file, "r") as f:
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
    aggr_dfs = []
    # Attributes to save
    # num_clients
    # num_byz_nodes
    # learning_rate
    # damp_alpha
    iterx = 0
    for running_stats, cfg_data in outputs2:
        pass
        name = cfg_data["name"]
        min_cluster_size = 0
        if "min_cluster_size" in cfg_data["server_args"]:
            min_cluster_size = int(cfg_data["server_args"]["min_cluster_size"])
        interaction_events = running_stats[3]
        ie_df = pd.DataFrame(interaction_events, columns=["client_id", "wall_time", "min_ct", "client_ct"])
        ie_df["alg"] = name
        ie_df["round"] = ie_df.index
        interaction_dfs.append(ie_df)
        local_df = pd.DataFrame(running_stats[0], columns=["round", "accuracy", "loss", "backdoor_accuracy"])
        parts = name.split("-")[-1].split("_")
        agg_local_df = pd.DataFrame(running_stats[4], columns=["round", "wall-time"])
        agg_local_df["idx"] = agg_local_df.index
        # print(parts)
        # pp.pprint(cfg_data)

        server_lr = cfg_data["server_args"]["learning_rate"]
        num_rounds = cfg_data["num_rounds"]
        num_clients = cfg_data["clients"]["n"]
        num_byz_nodes = cfg_data["clients"]["f"]
        num_buffers = 0
        if "num_buffers" in cfg_data["server_args"]:
            num_buffers = cfg_data["server_args"]["num_buffers"]
        name_suffix = "-async"
        if "aggregation_bound" in cfg_data["server_args"]:
            name_suffix = "sync"
        kardam_damp = ""
        if "damp_alpha" in cfg_data["server_args"]:
            kardam_damp = cfg_data["server_args"]["damp_alpha"]
        f = int(parts[0][1:])
        if "disable_alpha" in cfg_data["server_args"]:
            disable_alpha = cfg_data["server_args"]["disable_alpha"]
        else:
            disable_alpha = False

        if "enable_scaling_factor" in cfg_data["server_args"]:
            enable_scaling_factor = cfg_data["server_args"]["enable_scaling_factor"]
        else:
            enable_scaling_factor = False

        if "impact_delayed" in cfg_data["server_args"]:
            impact_delayed = cfg_data["server_args"]["impact_delayed"]
        else:
            impact_delayed = 1.0
        alg_version = ''
        if 'alg_version' in cfg_data['server_args']:
            alg_version = '-'+ cfg_data["server_args"]["alg_version"]

        byz_type = "None"
        if f:
            byz_type = parts[-1].upper()
            byz_type = cfg_data["clients"]["f_type"]
        local_df["f"] = f
        local_df["iterx"] = f'{iterx}'

        local_df["byz_type"] = byz_type
        local_df["disable_alpha"] = disable_alpha
        local_df["enable_scaling_factor"] = enable_scaling_factor
        local_df["num_clients"] = num_clients
        local_df["impact_delayed"] = impact_delayed
        local_df["alg_name"] = parts[-2]
        # local_df['use_lipschitz_server_approx'] = cfg_data['server_args']['use_lipschitz_server_approx']
        local_df_name = f"{parts[-2]}-f{f}-id{impact_delayed}-esf{int(enable_scaling_factor)}-{name_suffix}{alg_version}"
        # print(local_df_name, parts)
        local_df["name"] = local_df_name
        ie_df["name"] = local_df_name
        agg_local_df["name"] = local_df_name

        ct = [[x, name, "clients", local_df_name] for x in cfg_data["clients"]["client_ct"]]
        ct += [[x, name, "f_clients", local_df_name] for x in cfg_data["clients"]["f_ct"]]
        local_client_dist_df = pd.DataFrame(ct, columns=["ct", "alg", "type", "name"])
        client_dist_dfs.append(local_client_dist_df)
        aggr_dfs.append(agg_local_df)

        dfs.append(local_df)
        iterx += 1

    # pp.pprint(cfg_data)
    # print(num_byz_nodes)
    # exit()

    # Plots
    # 0 byz nodes
    # cols = algorithm
    # rows num_nodes

    server_df = pd.concat(dfs, ignore_index=True)
    client_dist_df = pd.concat(client_dist_dfs, ignore_index=True)
    interaction_events_df = pd.concat(interaction_dfs, ignore_index=True)
    aggregation_events_df = pd.concat(aggr_dfs, ignore_index=True)


    print(server_df.columns)

    print(server_df.groupby(['alg_name', 'num_clients', 'f']).mean().reset_index())

    print(server_df['f'].unique())
    print(server_df['iterx'].unique())
    print(server_df.groupby(['iterx']).count().reset_index())


    # print(server_df.columns)
    # for idx, row in server_df[server_df['alg_name'] == 'Kardam'][['round', 'alg_name', 'backdoor_accuracy', 'accuracy']].iterrows():
    # # for idx, row in server_df.groupby(['alg_name', 'round']).median().reset_index()[['alg_name', 'backdoor_accuracy', 'accuracy']].iterrows():
    #     print(row.values)

    # print(server_df[server_df['alg_name'] == 'Kardam'].max())
    # # exit()
    # algs = list(server_df['alg_name'].unique())
    # for alg_i in algs:
    #     plt.figure(figsize=(24,16))

    #     sns.lineplot(data=server_df[server_df['alg_name'] == alg_i], x='round', y='backdoor_accuracy', hue='iterx')

    #     plt.savefig(f'backdoor-{alg_i}.png')


    # print(server_df.groupby(['iterx']).max().reset_index().groupby('alg_name').mean().reset_index())
    # exit()


    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5})  # type: ignore
    fig_size = (12, 6)

    # for idx, row in interaction_events_df.iterrows():
    #     print(row)

    # graph_file = graphs_path / f"{exp_name}_wall_time.png"

    # plt.figure()
    # sns.lineplot(data=interaction_events_df, x='wall_time', y='round', hue='name')
    # plt.savefig(graph_file)
    # plt.show()

    # exit()
    aggregation_events_df = aggregation_events_df.groupby(["name", "idx"]).mean().reset_index()

    print(aggregation_events_df.groupby("name").count())

    plt.figure()
    graph_file = graphs_path / f"{exp_name}_aggregation_stats.png"
    print(f"Generating plot: {graph_file}")
    g = sns.lineplot(
        data=aggregation_events_df, x="idx", y="round", hue="name", style="name", markers=True, dashes=False
    )
    plt.savefig(graph_file)
    print(aggregation_events_df.columns)
    plt.figure()
    graph_file = graphs_path / f"{exp_name}_aggregation_stats_wall_time.png"
    print(f"Generating plot: {graph_file}")
    g = sns.lineplot(
        data=aggregation_events_df, x="wall-time", y="round", hue="name", style="name", markers=True, dashes=False
    )
    plt.savefig(graph_file)

    for n_byz, byz_type in itertools.product(server_df["f"].unique(), server_df["byz_type"].unique()):
        s_df = server_df[(server_df["f"] == n_byz) & (server_df["byz_type"] == byz_type)]
        graph_file = graphs_path / f"{exp_name}_b{n_byz}_t{byz_type}_rounds.png"
        print(f"Generating plot: {graph_file}")

        local_df = s_df
        if len(local_df):
            plt.figure(figsize=fig_size)
            g = sns.lineplot(data=local_df, x="round", y="accuracy", hue="name")
            # g = sns.FacetGrid(local_df, col="alg_name",  row="num_clients", hue='use_lipschitz_server_approx', aspect=2)
            # g.map(sns.lineplot, "round", "accuracy")
            # g.add_legend()
            plt.savefig(graph_file)
        else:
            print(f"Not plotting due to empty dataframe")

        merged = pd.merge(left=s_df, right=interaction_events_df, on=["round", "name"], how="left")
        # print(merged.columns)

        graph_file = graphs_path / f"{exp_name}_b{n_byz}_wall_time.png"
        print(f"Generating plot: {graph_file}")
        local_df = merged

        print(local_df.columns)
        print(local_df.head())
        # Find maximum value of wall_time
        print('Devider')
        print(local_df.groupby('name').max())
        # Find shortest interval between wall_time when grouped by iterx
        print(local_df[['wall_time', 'iterx', 'name']].groupby(['iterx', 'name']).diff().min())

        # Convert wall_time to string
        local_df['wall_time'] = local_df['wall_time'].astype(str)

        # Convert wall_time to datetime with format %S
        # local_df['Time'] = pd.to_datetime(local_df['wall_time'], format='%S').dt.time
        local_df['time'] = pd.to_datetime(local_df['wall_time'], unit='ms')

        print(local_df.head(10))
        # Convert wall_time to float
        local_df['wall_time'] = local_df['wall_time'].astype(float)

        
        print('Breaker')

        # Resample to 1 second intervals for each iterx
        local_df = local_df.set_index('time').groupby(['iterx', 'name']).resample('1S').mean().reset_index()
        # Convert column time to seconds
        local_df['time'] = local_df['time'].dt.second * 1000

        # Replace the value of name to Flame if the name starts with Flame
        local_df['name'] = local_df['name'].apply(lambda x: 'Flame' if x.startswith('Flame') else x)
        # Replace the value of name to Catalyst if the name starts with SemiAsync
        local_df['name'] = local_df['name'].apply(lambda x: 'Catalyst' if x.startswith('SemiAsync') else x)


        # Ask the user for yes or no to save the data to a csv file
        answer = input("Save data to csv?")
        # Get current file path
        out_path = Path(__file__).parent.parent.parent / 'data-processing' / 'data'
        print(out_path.absolute())
        if answer.upper() in ["Y", "YES"]:
            file_loc = out_path / f'wall_time_catalyst_vs_flame.csv'
            print('Saving data to csv')
            local_df.to_csv(file_loc)
            # Print the file location
        else:
            print('Not saving data to csv')
    
        print(local_df.head(10))
        if len(local_df):
            plt.figure(figsize=fig_size)
            g = sns.lineplot(data=local_df, x="time", y="accuracy", hue="name", errorbar=('ci', 50))

            # g = sns.FacetGrid(local_df, col="alg_name",  row="num_clients", hue='use_lipschitz_server_approx', aspect=2)
            # g.map(sns.lineplot, "round", "accuracy")
            # Remove the legend title but keep the legend and the entries
            if g.legend_:
                g.legend_.set_title(None)
                
            # g.add_legend()
            # Max x value is 50000
            # Change the legend to the name of the algorithm
            
            plt.xlim(0, 50000)
            plt.savefig(graph_file)
        else:
            print(f"Not plotting due to empty dataframe")

    # inspect = merged[merged["name"] == "PessimisticServer-f0-0.1--sync-1"]
    # # inspect = interaction_events_df[interaction_events_df['name']=='PessimisticServer-f0-0.1--sync-1']
    # inspect = inspect.sort_values(by=["wall_time"])
    # for idx, row in inspect.iterrows():
    #     # print(f'[{idx} | {row["client_id"]}] {row["round"]} :: {row["wall_time"]} :: {row["name"]} :: {row["client_ct"]}')
    #     print(
    #         f'[{idx} | {row["client_id"]}] {row["round"]} ::{row["accuracy"]} :: {row["wall_time"]} :: {row["name"]} :: {row["client_ct"]}'
    #     )
    # print(inspect.columns)

    # for idx, row in merged['name'].unique() == ''

    # graph_file = graphs_path / f'{exp_name}_general.png'
    # print(f'Generating plot: {graph_file}')
    # local_df = server_df[server_df['f'] == 0]
    # plt.figure(figsize=fig_size)
    # g = sns.FacetGrid(local_df, col="name",  row="num_clients", hue='use_lipschitz_server_approx', aspect=2)
    # g.map(sns.lineplot, "round", "accuracy")
    # g.add_legend()
    # plt.savefig(graph_file)
    # last_round = server_df.max()['round']
    # single_interaction_df = interaction_events_df
    # print(single_interaction_df)
    plt.figure()
    graph_file = graphs_path / f"{exp_name}_client_kde.png"

    sns.kdeplot(data=client_dist_df, x="ct", hue="name")
    plt.title("Compute kde")
    plt.savefig(graph_file)
    print(f"Saving figure at {graph_file}")

    plt.show()

    exit()

    local_df = server_df[server_df["num_clients"] == 100]

    graph_file = graphs_path / f"{exp_name}_byz.png"
    print(f"Generating plot: {graph_file}")
    # Visualize data
    fig = plt.figure(figsize=fig_size)
    g = sns.lineplot(data=local_df, x="round", y="accuracy", hue="name", errorbar=("ci", 95))
    plt.title(f"Algorithms Flameserver")
    plt.xlabel("Rounds")
    plt.ylabel("Test accuracy")
    if g.legend_:
        g.legend_.set_title(None)
    plt.savefig(graph_file)
    print(f"Saving figure at {graph_file}")

    if show_plots:
        plt.show()
    plt.close(fig)

    single_interaction_df = interaction_events_df
    # print(single_interaction_df)

    plt.figure()
    graph_file = graphs_path / f"{exp_name}_client_kde.png"

    sns.kdeplot(data=single_interaction_df, x="client_ct")
    plt.title("Compute kde")
    plt.savefig(graph_file)
    print(f"Saving figure at {graph_file}")

    plt.show()

    # Time based statistics
    # print(single_interaction_df)
    # print(local_df)

    combined_df = local_df.merge(
        single_interaction_df, how="left", left_on=["name", "round"], right_on=["name", "round"]
    )
    local_df["wall_time"] = single_interaction_df["wall_time"]
    print(combined_df["name"].unique())

    graph_file = graphs_path / f"{exp_name}_wall_time.png"
    fig = plt.figure(figsize=fig_size)
    g = sns.lineplot(data=combined_df, x="wall_time", y="accuracy", hue="name", errorbar=("ci", 95))
    plt.title(f"Algorithms Flameserver")
    plt.xlabel("Time (s)")
    plt.ylabel("Test accuracy")
    if g.legend_:
        g.legend_.set_title(None)
    plt.savefig(graph_file)
    print(f"Saving figure at {graph_file}")
    plt.show()
    plt.close(fig)
