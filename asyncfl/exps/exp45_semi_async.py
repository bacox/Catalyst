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

if __name__ == "__main__":
    args = cli_options()

    print("Exp 45: Semi Async server")

    exp_name = "exp45_semi_async"

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
        pool_size = 1
        configs = []
        # model_name = 'cifar10-resnet9'
        # model_name = 'cifar10-resnet18'
        # model_name = 'cifar10-lenet'
        # dataset = 'cifar10'
        model_name = "mnist-cnn"
        dataset = "mnist"
        # num_byz_nodes = [0, 1, 3]
        # num_byz_nodes = [1]
        # num_byz_nodes = [0]
        num_rounds = 3
        idx = 1 # Most likely should not be changed in most cases
        repetitions = 1
        exp_id = 0
        # server_lr = 0.005
        server_lr = 0.1
        # num_clients = 50
        # num_clients = 10

        var_sets = [
            # {"num_clients": 40, "num_byz_nodes": 0, "flame_hist": 3},
            # {"num_clients": 40, "num_byz_nodes": 0, "flame_hist": 3},
            {"num_clients": 40, "num_byz_nodes": 10, "flame_hist": 3},
        ]

        attacks = [
            # [AFL.NGClient, {"magnitude": 10, "sampler": "uniform", "sampler_args": {}}],
            [AFL.RDCLient, {'a_atk':0.1, 'sampler': 'uniform', 'sampler_args': {}}],
        ]

        servers = [
            # [
            #     AFL.Kardam,
            #     {
            #         "learning_rate": server_lr,
            #         "damp_alpha": 0.1,
            #         "use_fedasync_alpha": False,
            #         "use_fedasync_aggr": True,
            #         "use_lipschitz_server_approx": False,
            #     },
            #     'async'
            # ],
            # # [AFL.PessimisticServer, {"learning_rate": server_lr, "k": 3, "disable_alpha": True}, 'semi-async'],
            # [AFL.FedAsync,{'learning_rate': server_lr},'async'],

            [AFL.SemiAsync, {"learning_rate": server_lr, "k": 4, "aggregation_bound": 10, "disable_alpha": True}, 'semi-async'],
            # [
            #     AFL.PessimisticServer,
            #     {"learning_rate": server_lr, "k": 3, "aggregation_bound": 40, "disable_alpha": False},
            #     'semi-async'
            # ],
            # # [AFL.SaSGD,{'learning_rate': server_lr}],
            # # [AFL.Server,{'learning_rate': server_lr}],
            # # [AFL.FedAsync,{'learning_rate': 0.05},],
            # # [AFL.FedAsync,{'learning_rate': 0.01},],
            # # [AFL.FedWait,{'learning_rate': server_lr}],
            # # [AFL.Server,{'learning_rate': server_lr}],
            # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': 20, 'aggr_mode': 'trmean'},'async'],
            # # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': 15, 'aggr_mode': 'trmean'},'async'],
            # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': 15},'async'],
            # [AFL.BASGD,{'learning_rate': server_lr, 'num_buffers': 10, 'aggr_mode': 'trmean'},'async'],
        ]

        f0_keys = []
        generated_ct = {}

        # @TODO: BASGD alg works with gradients. In the implementation we use weights. This is a difference.

        
        for _r, server, var_set, atk in itertools.product(range(repetitions), servers, var_sets, attacks):
            num_clients, f, fh = var_set.values()
            ct_key = f"{num_clients}-{f}-{fh}"
            # print(n, f, fh)
            # ct_key = f'{num_clients}-{f}'
            if ct_key not in generated_ct.keys():
                ct_clients = np.abs(np.random.normal(100, 20.0, num_clients - f))
                f_ct = np.abs(np.random.normal(100, 20.0, f))
                generated_ct[ct_key] = [ct_clients, f_ct]
            ct_clients, f_ct = copy.deepcopy(generated_ct[ct_key])
            print(ct_clients)

            # @TODO: Make sure they have exactly the same schedule!!
            # 1. Generate the same client schedule
            # 2. Generate the same data distribution
            #
            # Q? Can this be fixed internally by having the same seed?

            # for _r, f, server, atk in itertools.product(range(repetitions), num_byz_nodes, servers, attacks):
            #     ct_key = f'{num_clients}-{f}'
            #     if ct_key not in generated_ct.keys():
            #         ct_clients = np.abs(np.random.normal(100, 40.0, num_clients - f))
            #         f_ct = np.abs(np.random.normal(100, 40.0, f))
            #         generated_ct[ct_key] = [ct_clients, f_ct]
            #     ct_clients, f_ct = copy.deepcopy(generated_ct[ct_key])

            # Round robin
            # f_ct = [1] * f
            # ct_clients = [1] * (num_clients - f)

            server_name = server[0].__name__
            attack_name = atk[0].__name__
            key_name = f'f{f}_n{num_clients}_lr{server_lr}_{model_name.replace("-", "_")}_{server_name}'
            if server[0] == AFL.FlameServer:
                key_name += f'_{server[1]["hist_size"]}'
            else:
                key_name += f"_0"
            exp_id += 1

            rounds = num_rounds
            if server[2] != 'semi-async':
                rounds = num_rounds * num_clients

            configs.append(
                {
                    "exp_id": exp_id,
                    "aggregation_type": server[2],
                    "client_participartion": 0.2,
                    "name": f"{server_name}-async-{key_name}",
                    "num_rounds": rounds,
                    "client_batch_size": -1,
                    "eval_interval": 5,
                    "clients": {
                        "client": AFL.Client,
                        "client_args": {"learning_rate": server_lr, "sampler": "limitlabel", "sampler_args": (7, 42)},
                        "client_ct": ct_clients,
                        "n": num_clients,
                        "f": f,
                        # 'f': f,
                        "f_type": atk[0],
                        "f_args": atk[1],
                        "f_ct": f_ct,
                    },
                    "server": server[0],
                    "server_args": server[1],
                    "dataset_name": dataset,
                    "model_name": model_name,
                }
            )

        # for cfg in configs:
        #     print(dict_hash(cfg, exclude_keys=['exp_id']))

        # exit(0)
        # Run all experiments multithreaded
        # tmp_df = pd.DataFrame(ct_clients, columns=['ct'])
        # plt.figure()
        # graph_file = graphs_path / f'{exp_name}_client_kde.png'

        # sns.kdeplot(data=tmp_df, x='ct')
        # plt.title('Compute kde')
        # plt.savefig(graph_file)
        # print(f'Saving figure at {graph_file}')

        # plt.show()
        # exit()
        outputs = AFL.Scheduler.run_multiple(
            configs,
            pool_size=pool_size,
            outfile=data_file,
            clear_file=not args.autocomplete,
            multi_thread=multi_thread,
            autocomplete=args.autocomplete,
        )

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

    # Attributes to save
    # num_clients
    # num_byz_nodes
    # learning_rate
    # damp_alpha
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
        local_df = pd.DataFrame(running_stats[0], columns=["round", "accuracy", "loss"])
        parts = name.split("-")[-1].split("_")

        # print(parts)
        # pp.pprint(cfg_data)

        server_lr = cfg_data["server_args"]["learning_rate"]
        num_rounds = cfg_data["num_rounds"]
        num_clients = cfg_data["clients"]["n"]
        num_byz_nodes = cfg_data["clients"]["f"]
        num_buffers = 0
        if 'num_buffers' in cfg_data['server_args']:
            num_buffers = cfg_data['server_args']['num_buffers']
        name_suffix = "-async"
        if "aggregation_bound" in cfg_data["server_args"]:
            name_suffix = "sync"
        kardam_damp = ""
        if "damp_alpha" in cfg_data["server_args"]:
            kardam_damp = cfg_data["server_args"]["damp_alpha"]
        f = int(parts[0][1:])
        byz_type = "None"
        if f:
            byz_type = parts[-1].upper()
        local_df["f"] = f
        local_df["byz_type"] = byz_type
        local_df["num_clients"] = num_clients
        local_df["alg_name"] = parts[-2]
        # local_df['use_lipschitz_server_approx'] = cfg_data['server_args']['use_lipschitz_server_approx']
        local_df_name = f"{parts[-2]}-f{f}-{server_lr}-{num_buffers}-{name_suffix}"
        # print(local_df_name, parts)
        local_df["name"] = local_df_name
        ie_df["name"] = local_df_name

        ct = [[x, name, "clients", local_df_name] for x in cfg_data["clients"]["client_ct"]]
        ct += [[x, name, "f_clients", local_df_name] for x in cfg_data["clients"]["f_ct"]]
        local_client_dist_df = pd.DataFrame(ct, columns=["ct", "alg", "type", "name"])
        client_dist_dfs.append(local_client_dist_df)

        dfs.append(local_df)

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

    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5})  # type: ignore
    fig_size = (12, 6)


    for idx, row in interaction_events_df.iterrows():
        print(row)
    
    graph_file = graphs_path / f"{exp_name}_wall_time.png"

    plt.figure()
    sns.lineplot(data=interaction_events_df, x='wall_time', y='round', hue='name')
    plt.savefig(graph_file)
    plt.show()

    # exit()


    for n_byz in server_df['f'].unique():
        s_df = server_df[server_df['f'] == n_byz]
        graph_file = graphs_path / f"{exp_name}_b{n_byz}_rounds.png"
        print(f"Generating plot: {graph_file}")
        local_df = s_df
        plt.figure(figsize=fig_size)
        g = sns.lineplot(data=local_df, x="round", y="accuracy", hue="name")
        # g = sns.FacetGrid(local_df, col="alg_name",  row="num_clients", hue='use_lipschitz_server_approx', aspect=2)
        # g.map(sns.lineplot, "round", "accuracy")
        # g.add_legend()
        plt.savefig(graph_file)

        merged = pd.merge(left=s_df, right=interaction_events_df, on=["round", "name"], how="left")
        # print(merged.columns)

        graph_file = graphs_path / f"{exp_name}_b{n_byz}_wall_time.png"
        print(f"Generating plot: {graph_file}")
        local_df = merged
        plt.figure(figsize=fig_size)
        g = sns.lineplot(data=local_df, x="wall_time", y="accuracy", hue="name")
        # g = sns.FacetGrid(local_df, col="alg_name",  row="num_clients", hue='use_lipschitz_server_approx', aspect=2)
        # g.map(sns.lineplot, "round", "accuracy")
        # g.add_legend()
        plt.savefig(graph_file)

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
