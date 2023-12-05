import json
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class ResultDataFrames:
    server_df: pd.DataFrame
    interaction_events_df: pd.DataFrame
    client_dist_df: pd.DataFrame
    metric: Literal["Accuracy", "Perplexity"]


def prepare_dfs(data_file: Path) -> ResultDataFrames:
    outputs = []
    metric = "Accuracy"
    server_dfs = []
    interaction_dfs = []
    client_dist_dfs = []

    with open(data_file, "r") as f:
        outputs = json.load(f)

    for running_stats, cfg_data in outputs:
        name = cfg_data["name"]
        metric = "Perplexity" if "lstm" in cfg_data["model_name"] else "Accuracy"
        name_parts = name.split("_")
        alg_name = name_parts[0]

        local_server_df = pd.DataFrame(running_stats[0], columns=["Round", metric, "loss"])
        f = cfg_data["clients"]["f"]
        local_server_df["f"] = f
        local_server_df["n"] = cfg_data["clients"]["n"]
        local_server_df["byz_type"] = name_parts[2]
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

    return ResultDataFrames(server_df, interaction_events_df, client_dist_df, metric)


def save_lineplots(res_dfs: ResultDataFrames, graphs_path: Path, exp_name: str) -> None:
    sns.set_theme(style="white", palette="Dark2", font_scale=1.5, rc={"lines.linewidth": 2.5})  # type: ignore
    fig_size = (12, 6)

    # for idx, row in interaction_events_df.iterrows():
    #     print(row)

    # graph_file = graphs_path / f"{exp_name}_walltime.png"
    # plt.figure()
    # sns.lineplot(data=res_dfs.interaction_events_df, x="Wall Time", y="Round", hue="alg")
    # print(f"Saving figure at {graph_file}")
    # plt.savefig(graph_file, bbox_inches="tight")

    s_df_groupby = res_dfs.server_df.groupby(["byz_type", "f", "n"], as_index=False, sort=False)

    for (byz_type, f, n), s_df in s_df_groupby:
        byz_type_norm = byz_type.translate(
            str.maketrans({" ": "", "\N{GREEK SMALL LETTER ALPHA}": "a"}))
        fname_prefix = f"{exp_name}_{byz_type_norm}_f{f}_n{n}"

        graph_file = graphs_path / f"{fname_prefix}_rounds.png"
        print(f"Generating plot: {graph_file}")
        plt.figure(figsize=fig_size)
        g = sns.lineplot(data=s_df, x="Round", y=res_dfs.metric, hue="alg")
        g.get_legend().set_title(None)
        plt.savefig(graph_file, bbox_inches="tight")

        merge_on = ["Round", "name", "alg", "exp_id"]
        merged = pd.merge(left=s_df, right=res_dfs.interaction_events_df, on=merge_on, how="left")

        graph_file = graphs_path / f"{fname_prefix}_walltime.png"
        print(f"Generating plot: {graph_file}")
        plt.figure(figsize=fig_size)
        g = sns.lineplot(data=merged, x="Wall Time", y=res_dfs.metric, hue="alg")
        g.get_legend().set_title(None)
        plt.savefig(graph_file, bbox_inches="tight")

    # plt.figure()
    # graph_file = graphs_path / f"{exp_name}_client_kde.png"
    # sns.kdeplot(data=res_dfs.client_dist_df, x="ct", hue="name")
    # plt.title("Compute kde")
    # print(f"Saving figure at {graph_file}")
    # plt.savefig(graph_file, bbox_inches="tight")


def fill_table(res_dfs: ResultDataFrames, timestamp: int) -> None:
    name_to_vals = defaultdict(list)
    groubpy = res_dfs.server_df.groupby(["byz_type", "alg", "exp_id"], as_index=False, sort=False)

    for (byz_type, alg, _exp_id), s_df in groubpy:
        merge_on = ["Round", "name", "alg", "exp_id"]
        merged = pd.merge(left=s_df, right=res_dfs.interaction_events_df, on=merge_on, how="left")
        k = f"{alg} | {byz_type}"
        filtered = merged[merged["Wall Time"] >= timestamp]
        if filtered.empty:
            name_to_vals[k].append(merged[res_dfs.metric].ffill().iloc[-1])  # TODO consider other options
        else:
            name_to_vals[k].append(filtered[res_dfs.metric].iloc[0])  # assumes timestamps were sorted

    # for k, v in name_to_vals.items():
    #     print(k, v)

    # print()

    name_to_acc_mean = {k: sum(v) / len(v) for k, v in name_to_vals.items()}
    name_to_acc_sdev = {
        k: sqrt(sum((x - name_to_acc_mean[k]) ** 2 for x in v) / len(v))
        for k, v in name_to_vals.items()
    }

    name_to_acc_mean = {k: round(v, 1) for k, v in name_to_acc_mean.items()}
    name_to_acc_sdev = {k: round(v, 1) for k, v in name_to_acc_sdev.items()}

    for k in name_to_vals:
        print(k, name_to_acc_mean[k], name_to_acc_sdev[k])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("exp_name", help="Experiment name")
    parser.add_argument("-l", help="Line plots", action="store_true")
    parser.add_argument("-t", help="Table threshold", type=int, default=0)
    args = parser.parse_args()

    (graphs_path := Path("graphs") / args.exp_name).mkdir(exist_ok=True, parents=True)

    res_dfs= prepare_dfs(Path(".data") / f"{args.exp_name}.json")

    if args.l:
        save_lineplots(res_dfs, graphs_path, args.exp_name)

    if args.t:
        fill_table(res_dfs, args.t)


if __name__ == "__main__":
    main()
