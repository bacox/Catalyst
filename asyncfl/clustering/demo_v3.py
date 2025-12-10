from typing import Dict
import numpy as np
import torch
from collections import Counter
import hdbscan
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from asyncfl.flame import flame_v3

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
    print(f'Selector: {-stored_rounds*clients_per_round:}')
    labels, major_label = _clustering(stored_gradients[-stored_rounds*clients_per_round:])
    # labels[-clients_per_round:] are the clustering labels of the last round
    majority_clients_index = np.where(labels[-clients_per_round:] == major_label)[0]
    # get index of candidates clients
    candidates_clients = client_index[-clients_per_round:][majority_clients_index]
    return majority_clients_index, candidates_clients, labels

def _clustering(stored_gradients):
    # transform the stored gradients into a numpy array
    stored_gradients = np.concatenate(stored_gradients, axis=0)
    #clf = DBSCAN(eps=0.5, min_samples=5).fit(stored_gradients)
    clf = hdbscan.HDBSCAN(min_cluster_size=3).fit(stored_gradients)
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


if __name__ == '__main__':
    #test the clustering function
    print('Starting clustering demo from Chaoyi')
    client_index = ['client8', 'client8', 'client1', 'client2', 'client5', 'client6', 'client3', 'client1']
    stored_gradients_minor = [torch.zeros(1, 2) for _  in range(3)]
    stored_gradients_major = [torch.ones(1, 2) for _  in range(5)]
    stored_gradients_1 = stored_gradients_minor + stored_gradients_major
    stored_gradients_2 = stored_gradients_major + stored_gradients_minor
    stored_gradients_1_n = [x.squeeze().cpu().numpy() for x in stored_gradients_1]
    stored_gradients_2_n = [x.squeeze().cpu().numpy() for x in stored_gradients_2]
    clustered = flame_v3(stored_gradients_1_n[2:6], stored_gradients_1_n[0], 3)
    print('input:')
    [print(x) for x in stored_gradients_1_n[2:6]]
    print('output')
    [print(x) for x in clustered]
    print('next')
    clustered = flame_v3(stored_gradients_2_n, stored_gradients_2_n[0], 3)
    [print(x) for x in clustered]
    # print(stored_gradients_1)
    # majority, candidates, labels1 = clustering(stored_gradients_1, client_index, clients_per_round=4, stored_rounds=1)
    # print("clients in the largest cluster: ", majority)
    # print("candidates in this round: ", candidates)

    # print(stored_gradients_2)
    # majority, candidates, labels2 = clustering(stored_gradients_2, client_index, clients_per_round=4, stored_rounds=1)
    # print("clients in the largest cluster: ", majority)
    # print("candidates in this round: ", candidates)

    # numpy_grads = np.concatenate(stored_gradients_1[-len(labels1):], axis=0)
    # df = pd.DataFrame(numpy_grads, columns=['x', 'y'])
    # df['labels'] = labels1

    # plt.figure()
    # sns.scatterplot(data=df, x='x', y='y', hue='labels')
    # plt.title(f"stored_gradients_1")
    # plt.show()


    # numpy_grads = np.concatenate(stored_gradients_2[-len(labels2):], axis=0)
    # df = pd.DataFrame(numpy_grads, columns=['x', 'y'])
    # df['labels'] = labels2

    # plt.figure()
    # sns.scatterplot(data=df, x='x', y='y', hue='labels')
    # plt.title(f"stored_gradients_2")
    # plt.show()

