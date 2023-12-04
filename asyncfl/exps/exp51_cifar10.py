import copy
from pathlib import Path
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)
import itertools
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import asyncfl as AFL
from asyncfl.util import cli_options

# Turn interactive plotting off
plt.ioff()

if __name__ == "__main__":
    args = cli_options()

    print("Exp 51: CIFAR10")

    exp_name = "exp51_cifar10"

    (data_path := Path(".data")).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path("graphs") / exp_name).mkdir(exist_ok=True, parents=True)
    data_file = data_path / f"{exp_name}.json"

    if not args.o:
        multi_thread = True
        pool_size = 1
        model_name = "cifar10-resnet18"
        dataset = "cifar10"
        num_rounds = 80  # TODO: change to 140?
        repetitions = 3
        lr_all = 0.1

        var_sets = [
            {
                "num_clients": 40,
                "num_byz_nodes": 0,
                "ct_skew": 20
            },
            {
                "num_clients": 40,
                "num_byz_nodes": 10,
                "ct_skew": 20
            },
        ]

        client_args = {
            "learning_rate": lr_all,
            "sampler": "limitlabel",
            "sampler_args": (7, 42)
        }

        attacks = [
            [
                AFL.RDCLient,
                {
                    **client_args,
                    "a_atk": 0.1,
                }
            ],
            [
                AFL.NGClient,
                {
                    **client_args,
                    "magnitude": 10
                }
            ]
        ]

        servers = [
            [
                AFL.FedAsync,
                {
                    "learning_rate": lr_all,
                    "mitigate_staleness": True
                },
                "async"
            ],
            [
                AFL.Kardam,
                {
                    "learning_rate": lr_all,
                    "damp_alpha": 0.1,
                    "use_fedasync_alpha": False,
                    "use_fedasync_aggr": True,
                    "use_lipschitz_server_approx": False
                },
                "async"
            ],
            [
                AFL.BASGD,
                {
                    "learning_rate": lr_all,
                    "num_buffers": lambda f, _n: 2 * f + 1,
                    "aggr_mode": "median"
                },
                "async"
            ],
            [
                AFL.PessimisticServer,  # async
                {
                    "learning_rate": lr_all,
                    "k": 5,
                    "aggregation_bound": lambda f, _n: max(2, 2 * f + 1),
                    "disable_alpha": True,
                    "enable_scaling_factor": False,
                    "impact_delayed": 1.0
                },
                "semi-async"
            ],
            [
                AFL.PessimisticServer,
                {
                    "learning_rate": lr_all,
                    "k": 5,
                    "aggregation_bound": lambda _f, n: n,
                    "disable_alpha": True,
                    "enable_scaling_factor": False,
                    "impact_delayed": 1.0
                },
                "semi-async"
            ]
        ]

        generated_ct = {}
        configs = []
        idx = 1  # Most likely should not be changed in most cases
        exp_id = 0

        for _r, server, var_set, atk in itertools.product(range(repetitions), servers, var_sets, attacks):
            num_clients, f, ct_skew = var_set.values()
            ct_key = f"{num_clients}-{f}-{ct_skew}"
            if ct_key not in generated_ct.keys():
                ct_clients = np.abs(np.random.normal(100, ct_skew, num_clients - f))
                f_ct = np.abs(np.random.normal(100, ct_skew, f))
                generated_ct[ct_key] = [ct_clients, f_ct]
            ct_clients, f_ct = copy.deepcopy(generated_ct[ct_key])
            print(ct_clients)

            server_args = {k: v(f, num_clients) if callable(v) else v for k, v in server[1].items()}
            server_name = server[0].__name__
            if server_name == "FedAsync":
                pass
            elif server_name == "BASGD":
                server_name += f" (B={server_args['num_buffers']}, m={server_args['aggr_mode']})"
            elif server_name == "Kardam":
                server_name += f" (\N{GREEK SMALL LETTER GAMMA}={server_args['damp_alpha']})"
            elif server_name == "PessimisticServer":
                server_name = f"Catalyst (k={server_args['k']}, b={server_args['aggregation_bound']}, d={server_args['impact_delayed']})"
            attack_name = atk[0].__name__
            if attack_name == "RDCLient":
                attack_name = f"Random Perturbation (\N{GREEK SMALL LETTER ALPHA}={atk[1]['a_atk']})"
            elif attack_name == "NGClient":
                attack_name = f"Gradient Inversion (M={atk[1]['magnitude']})"
            key_name = f"{server_name}_{server[2]}_{attack_name}_f{f}_n{num_clients}_lr{lr_all}_{model_name}"
            exp_id += 1

            rounds = num_rounds
            if server[2] != "semi-async":
                rounds = num_rounds * num_clients

            configs.append(
                {
                    "exp_id": exp_id,
                    "aggregation_type": server[2],
                    "client_participartion": 0.2,
                    "name": key_name,
                    "num_rounds": rounds,
                    "client_batch_size": -1,
                    "eval_interval": 5,
                    "clients": {
                        "client": AFL.Client,
                        "client_args": client_args,
                        "client_ct": ct_clients,
                        "n": num_clients,
                        "f": f,
                        "f_type": atk[0],
                        "f_args": atk[1],
                        "f_ct": f_ct,
                    },
                    "server": server[0],
                    "server_args": server_args,
                    "dataset_name": dataset,
                    "model_name": model_name,
                }
            )

        outputs = AFL.Scheduler.run_multiple(
            configs,
            pool_size=pool_size,
            outfile=data_file,
            clear_file=not args.autocomplete,
            multi_thread=multi_thread,
            autocomplete=args.autocomplete,
        )

    # Load raw data from file
    outputs_ = ""
    with open(data_file, "r") as f:
        outputs_ = json.load(f)

    if not args.o:
        exit()

    show_plots = False
    if args.show_plots:
        show_plots = True

    # Process data into dataframe
    server_dfs = []
    interaction_dfs = []
    client_dist_dfs = []

    metric = ""

    for running_stats, cfg_data in outputs_:
        name = cfg_data["name"]
        metric = "Perplexity" if "lstm" in cfg_data["model_name"] else "Accuracy"
        name_parts = name.split("_")
        alg_name = name_parts[0]

        local_server_df = pd.DataFrame(running_stats[0], columns=["Round", metric, "loss"])
        f = cfg_data["clients"]["f"]
        local_server_df["f"] = f
        local_server_df["n"] = cfg_data["clients"]["n"]
        local_server_df["byz_type"] = name_parts[2] if f else "None"
        local_server_df["num_clients"] = cfg_data["clients"]["n"]
        local_server_df["alg"] = alg_name
        local_server_df["name"] = name
        local_server_df["exp_id"] = cfg_data["exp_id"]
        # local_server_df["ct_skew"] = cfg_data["ct_skew"]
        server_dfs.append(local_server_df)

        local_interaction_df = pd.DataFrame(running_stats[3], columns=["client_id", "Wall Time", "min_ct", "client_ct"])
        local_interaction_df["alg"] = alg_name
        local_interaction_df["Round"] = local_interaction_df.index
        local_interaction_df["name"] = name
        local_interaction_df["exp_id"] = cfg_data["exp_id"]
        interaction_dfs.append(local_interaction_df)

        ct = [[x, name, "clients", name] for x in cfg_data["clients"]["client_ct"]]
        ct += [[x, name, "f_clients", name] for x in cfg_data["clients"]["f_ct"]]
        local_client_dist_df = pd.DataFrame(ct, columns=["ct", "alg", "type", "name"])
        client_dist_dfs.append(local_client_dist_df)

    server_df = pd.concat(server_dfs, ignore_index=True)
    interaction_events_df = pd.concat(interaction_dfs, ignore_index=True)
    client_dist_df = pd.concat(client_dist_dfs, ignore_index=True)

    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5})  # type: ignore
    fig_size = (12, 6)

    # for idx, row in interaction_events_df.iterrows():
    #     print(row)

    graph_file = graphs_path / f"{exp_name}_walltime.png"
    plt.figure()
    sns.lineplot(data=interaction_events_df, x="Wall Time", y="Round", hue="alg")
    print(f"Saving figure at {graph_file}")
    plt.savefig(graph_file, bbox_inches="tight")
    # plt.show()

    for (byz_type, f, n), s_df in server_df.groupby(["byz_type", "f", "n"], as_index=False, sort=False):
        byz_type_norm = byz_type.translate(
            str.maketrans({" ": "", "\N{GREEK SMALL LETTER ALPHA}": "a"}))
        fname_prefix = f"{exp_name}_{byz_type_norm}_f{f}_n{n}"

        graph_file = graphs_path / f"{fname_prefix}_rounds.png"
        print(f"Generating plot: {graph_file}")
        plt.figure(figsize=fig_size)
        g = sns.lineplot(data=s_df, x="Round", y=metric, hue="alg")
        g.get_legend().set_title(None)
        plt.savefig(graph_file, bbox_inches="tight")

        merged = pd.merge(left=s_df, right=interaction_events_df, on=["Round", "name", "alg", "exp_id"], how="left")

        graph_file = graphs_path / f"{fname_prefix}_walltime.png"
        print(f"Generating plot: {graph_file}")
        plt.figure(figsize=fig_size)
        g = sns.lineplot(data=merged, x="Wall Time", y=metric, hue="alg")
        g.get_legend().set_title(None)
        plt.savefig(graph_file, bbox_inches="tight")

    plt.figure()
    graph_file = graphs_path / f"{exp_name}_client_kde.png"
    sns.kdeplot(data=client_dist_df, x="ct", hue="name")
    plt.title("Compute kde")
    print(f"Saving figure at {graph_file}")
    plt.savefig(graph_file, bbox_inches="tight")
    # plt.show()
