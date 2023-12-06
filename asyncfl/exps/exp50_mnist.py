import copy
from pathlib import Path
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np

import asyncfl as AFL
from asyncfl.util import cli_options

# Turn interactive plotting off
plt.ioff()

if __name__ == "__main__":
    args = cli_options()

    print("Exp 50: MNIST")

    exp_name = "exp50_mnist"

    (data_path := Path(".data")).mkdir(exist_ok=True, parents=True)
    (graphs_path := Path("graphs") / exp_name).mkdir(exist_ok=True, parents=True)
    data_file = data_path / f"{exp_name}.json"

    if not args.o:
        multi_thread = True
        pool_size = 6
        model_name = "mnist-cnn"
        dataset = "mnist"
        num_rounds = 30
        repetitions = 3
        lr_all = 0.05

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
                AFL.PessimisticServer,
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
            # [
            #     AFL.SemiAsync,
            #     {
            #         "learning_rate": lr_all,
            #         "k": 5,
            #         "aggregation_bound": lambda f, _n: max(2, 2 * f + 1),
            #         "disable_alpha": True,
            #     },
            #     "semi-async"
            # ]
        ]

        generated_ct = {}
        configs = []
        idx = 1  # Most likely should not be changed in most cases
        exp_id = 0

        for _r, server, var_set, atk in itertools.product(range(repetitions), servers, var_sets, attacks):
            num_clients, f, ct_skew = var_set.values()
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
            elif server_name == "SemiAsync":
                server_name = f"Semi-Synchronous (k={server_args['k']}, b={server_args['aggregation_bound']})"

            attack_name = atk[0].__name__
            if f < 1:
                attack_name = "None"
            elif attack_name == "RDCLient":
                attack_name = f"Random Perturbation (\N{GREEK SMALL LETTER ALPHA}={atk[1]['a_atk']})"
            elif attack_name == "NGClient":
                attack_name = f"Gradient Inversion (M={atk[1]['magnitude']})"
            key_name = f"{server_name}_{server[2]}_{attack_name}_f{f}_n{num_clients}_lr{lr_all}_{model_name}"
            if f < 1 and len([c for c in configs if c["name"] == key_name]) == repetitions:
                continue

            ct_key = f"{num_clients}-{f}-{ct_skew}"
            if ct_key not in generated_ct.keys():
                ct_clients = np.abs(np.random.normal(100, ct_skew, num_clients - f))
                f_ct = np.abs(np.random.normal(100, ct_skew, f))
                generated_ct[ct_key] = [ct_clients, f_ct]
            ct_clients, f_ct = copy.deepcopy(generated_ct[ct_key])
            print(ct_clients)

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
