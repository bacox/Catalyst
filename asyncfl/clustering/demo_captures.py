import sys
import numpy as np
import torch
from collections import Counter
import hdbscan
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from sklearn import svm
from sklearn.svm import OneClassSVM

from asyncfl.server import parameters_dict_to_vector_flt
logging.basicConfig(level=logging.INFO)


def clustering_2(stored_gradients, clients_per_round=10, stored_rounds = 10):
    labels, major_label = _clustering(stored_gradients)
    return major_label, labels[-1], labels


def clustering(stored_gradients, client_index, clients_per_round=10, stored_rounds=10):
    """
    Cluster the clients based on the stored gradients
    Args:
        stored_gradients:  history of gradients
        client_index: id of clients(gradients)
        clients_per_round: number of clients to be selected per round
        stored_rounds: number of global rounds to be considered for clustering
    Returns:
        aggregation_id: id of the clients to be aggregated
    """
    client_index = np.array(client_index)
    # labels are the cluster labels, major_label is the label of the majority cluster
    labels, major_label = _clustering(stored_gradients[-stored_rounds*clients_per_round:])
    logging.info(f'Using {-stored_rounds*clients_per_round}: much of history')
    logging.info(f'History: {stored_gradients[-stored_rounds*clients_per_round:]}')
    logging.info(f'Majority label: {major_label}')
    logging.info(f'Labels: {labels}')
    logging.info(f'np.where: {np.where(labels[-clients_per_round:] == major_label)[0]}')
    # labels[-clients_per_round:] are the clustering labels of the last round
    majority_clients_index = np.where(labels[-clients_per_round:] == major_label)[0]
    # get index of candidates clients
    candidates_clients = client_index[-clients_per_round:][majority_clients_index]
    return majority_clients_index, candidates_clients

def _one_class_svm(stored_gradients):
    sg = np.copy(stored_gradients)
    sg = np.concatenate(sg, axis=0)

    clf = OneClassSVM(nu=0.05, gamma=2)
    clf.fit(sg[:-1])
    # y_pred_train = clf.predict(X_train)
    # y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(sg)
    # print(f'One class labels:" {clf.labels_}')
    print(f'One class svm outlier: {y_pred_outliers}')

def _clustering(stored_gradients):
    _one_class_svm(stored_gradients)
    # transform the stored gradients into a numpy array
    stored_gradients = np.concatenate(stored_gradients, axis=0)
    # logging.info(f'SG: {stored_gradients}')
    #clf = DBSCAN(eps=0.5, min_samples=5).fit(stored_gradients)
    # clf = hdbscan.HDBSCAN(min_cluster_size=2, algorithm='best', alpha=1.0, approx_min_span_tree=True).fit(stored_gradients)
    clf = hdbscan.HDBSCAN(allow_single_cluster=True, min_samples=1).fit(stored_gradients)
    print(clf.outlier_scores_)
    threshold = pd.Series(clf.outlier_scores_).quantile(0.9)
    outliers = np.where(clf.outlier_scores_ > threshold)[0]
    print(f'Outliers: {outliers}')

    # clf.minimum_spanning_tree_.plot(edge_cmap='viridis',
    #                                   edge_alpha=0.6,
    #                                   node_size=80,
    #                                   edge_linewidth=2)
    # clf.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    logging.info(f'Labels: {clf.labels_}, grads: {stored_gradients}')
    major_label = _find_majority_label(clf)
    # get the clustering labels of stored gradients
    labels = clf.labels_
    return labels, major_label

def _find_majority_label(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    # major_id = set(major_id.reshape(-1))
    return major_label


def calculate_cluster_metrics(client_index, mal_index, candidates):
    y_true = [1 if i in mal_index else 0 for i in client_index]
    y_pred = [0 if i in candidates else 1 for i in client_index]
    # calculate the metrics
    # acc score
    acc = accuracy_score(y_true, y_pred)
    # precision score
    pre = precision_score(y_true, y_pred)
    # recall score
    rec = recall_score(y_true, y_pred)
    return acc, pre, rec


def _build_update(id: int, f_threshold, n_clients):
    std = .1
    mean = 0.
    t = torch.ones(1,2) * 10
    if id >= n_clients - f_threshold:
        t = torch.zeros(1, 2)
    # elif np.mod(id, 2) == 0:
    #     t = torch.ones(1,2) * -10
    return t + torch.randn(t.size()) * std + mean


def create_updates(schedule: List[int], f: int, n_clients):
    return [_build_update(x, f, n_clients) for x in schedule]


def compute_schedule(n: int, cpus: List[int]):
    return (list(range(len(cpus))) * n)[:n]


def do_round(client_id, candidate_gradient, history, round_id):
    history[client_id] = candidate_gradient
    grads_for_clustering = list(history.values())
    # grads_for_clustering = history + [candidate_gradient]
    grads_numpy = [x.numpy() for x in grads_for_clustering]
    # majority, candidate_label, labels = clustering_2(grads_numpy, 1, len(history))
    majority, candidate_label, labels = clustering_2(grads_numpy, 1, len(history.values()))
    print(f' maj={majority}, candidate={candidate_label} =?= {majority == candidate_label}')

    numpy_grads = np.concatenate(grads_for_clustering, axis=0)
    df = pd.DataFrame(numpy_grads, columns=['x', 'y'])
    df['labels'] = labels

    plt.figure(figsize=(2,2))
    sns.scatterplot(data=df, x='x', y='y', hue='labels')
    plt.title(f"Round {round_id}")
    plt.show()

    # x_hist = [x[0] for x in numpy_grads]
    # y_hist = [x[1] for x in numpy_grads]
    # plt.scatter(x_hist, y_hist, c=labels)
    # plt.legend(loc="upper left")
    # plt.title(f"Round {round_id}")
    # # plt.xlabel("Easting")
    # # plt.ylabel("Northing")
    # plt.show()
    # return grads_for_clustering, majority == candidate_label
    return history, majority == candidate_label


if __name__ == '__main__':
    print('Starting clustering demo')

    print(sys.argv[1])
    df = pd.read_json(sys.argv[1])
    # print(df)
    client_ids = df['client_id'].values
    client_weight_vectors = df['client_weight_vec'].values
    is_byz_vec= df['is_byzantine'].values
    round_ids = df.index.values
    # print(round_ids)
    round_ids = df.index.values
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cpu()
    num_users = 10
    malicious = 3
    frac = 1.0
    performance = []
    num_clients = max(int(frac * num_users), 1)
    num_malicious_clients = int(malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    for round, client_id, weight_vec, is_byzantine in zip(round_ids, client_ids, client_weight_vectors, is_byz_vec):
        # logging.info(f'[{round}]  {client_id}<byz={is_byzantine}>')
        if round < 2:
            continue
        local_model = client_weight_vectors[:round]

        cos_list=[]
        local_model_vector = []
        # size_cos_list = (2*malicious + 1)
        # size_cos_list = len(local_model)
        # size_cos_list = num_clients//2 + 1
        size_cos_list = 2*malicious
        
        for param in local_model[-size_cos_list:]:
            # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
            local_model_vector.append(torch.tensor(param))
        for i in range(len(local_model_vector)):
            cos_i = []
            for j in range(len(local_model_vector)):
                cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
                # cos_i.append(round(cos_ij.item(),4))
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)
        
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
        # print(clusterer.labels_)

        benign_client = []
        norm_list = np.array([])

        max_num_in_cluster=0
        max_cluster_index=0
        if clusterer.labels_.max() < 0:
            for i in range(len(local_model)):
                benign_client.append(i)
                # norm_list = np.append(norm_list,torch.norm(update_params[i],p=2).item())
        else:
            for index_cluster in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    benign_client.append(i)
        for i in range(len(local_model_vector)):
            # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
            # norm_list = np.append(norm_list,torch.norm(update_params[i]),p=2).item())  # no consider BN
            pass
        # logging.info(f'benign_client={benign_client}')
        logging.info(f'[{round}] Accept this update? {client_id in benign_client} is byzantine? {is_byzantine}, correct_decision? [{(client_id in benign_client) != is_byzantine}]')
        do_accept = (client_id in benign_client)
        performance.append([round, is_byzantine, client_id, do_accept, do_accept != is_byzantine])
        # for i in range(len(benign_client)):
        #     if benign_client[i] < num_malicious_clients:
        #         args.wrong_mal+=1
        #     else:
        #         #  minus per benign in cluster
        #         args.right_ben += 1
        # args.turn+=1
        # print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
        # print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    correct = 0
    incorrect_byz = 0
    correct_byz= 0
    correct_benign = 0
    incorrect_benign = 0
    for x in performance:
        if x[1]:
            if x[4]:
                correct_byz += 1
            else:
                incorrect_byz += 1
        else:
            if x[4]:
                correct_benign += 1
            else:
                incorrect_benign += 1
    # print(correct)
    # print(float(correct) / float(len(performance)))
    logging.info('='*20)
    logging.info(f'Benign:\tcorrect={correct_benign:5}, incorrect={incorrect_benign}')
    logging.info(f'Byz: \t\tcorrect={correct_byz:5}, incorrect={incorrect_byz}')
    exit(0)
    n_clients = 10
    cpus = [1] * n_clients
    client_index = [f'client_{x}' for x in list(range(n_clients))]
    f = 3
    num_rounds = 40
    schedule = compute_schedule(num_rounds, cpus)
    print(schedule)
    updates = create_updates(schedule, f, n_clients)

    history = []
    history_dict = {}

    for round_id, (update_grad, client_id) in enumerate(zip(updates, schedule)):
        if len(list(history_dict.values())) < 2:
            history_dict[client_id] = update_grad
            continue
        # if len(history) < 2:
        #     history += [update_grad]
        #     continue
        # print(f'Round {round_id} with client {client_id}')
        logging.info(f'{"="*10}')

        # history, accepted = do_round(client_id, update_grad, history, round_id)
        history_dict, accepted = do_round(client_id, update_grad, history_dict, round_id)
        logging.info(f'[Round {round_id}] Client {client_id}, <{update_grad}> accepted={accepted}')

    # majority, candidates = clustering(updates, client_index, clients_per_round=1, stored_rounds=20)
    # print(majority)
    # print(candidates)
    # print(updates)
    # for up in updates:
    #     print(up)