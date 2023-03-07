import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tqdm import tqdm

def create_client(id: int, ct=1):
    return {
        'value': np.random.randint(0, 1000),
        'id':  id,
        'picked': 0,
        'age': 0,
        'ct': ct,
        'ct_left': ct,
    }


def update(central_model, client):
    staleness = central_model['age'] - client['age']
    central_model['age'] += 1
    central_model['value'] += client['value']

    central_model['approx'] = central_model['value'] / central_model['age']
    central_model['difference'] = abs(central_model['true_avg'] - central_model['approx'])
    client['picked'] += 1
    
    client['age'] = central_model['age']
    return central_model, client, staleness


def run():
    print('Starting')

    num_clients = 100

    clients = [create_client(x) for x in range(num_clients)]

    avg_value = np.mean([x['value'] for x in clients])

    central_model = {
        'value': 0.0,
        'age': 0,
        'approx': 0,
        'difference': 0,
        'true_avg': avg_value
    }


    print(clients)

    print('Training')
    num_rounds = 10000
    data = []
    for round in tqdm(range(num_rounds)):

        rc = np.random.choice(clients)
        central_model, rc, staleness = update(central_model, rc)
        clients[rc['id']] = rc
        # print(rc)
        data.append([central_model['age'], central_model['approx'], 'approx', staleness])
        data.append([central_model['age'], avg_value, 'avg', 0])
    

    df = pd.DataFrame(data, columns=['age', 'value', 'type', 'staleness'])
    plt.figure()

    sns.lineplot(data=df, x='age', y='value', hue='type')

    plt.show()


    plt.figure()
    local_df = df[df['type']=='approx']
    sns.displot(data=local_df, x="staleness")
    plt.show()





def run_with_compute_time():
    num_clients = 1000
    num_rounds = 100000
    mu, sigma = 15, 3
    uniform_vals = [1, 20]
    exp_val = 1
    compute_times = np.random.normal(mu, sigma, num_clients)
    # compute_times = np.random.exponential(exp_val, num_clients)
    # compute_times = np.random.uniform(*uniform_vals, num_clients)

    df_ct = pd.DataFrame(compute_times, columns=['ct'])

    plt.figure()
    sns.displot(data=df_ct, x='ct', bins=20)
    plt.show()

    clients = [create_client(x[0], x[1]) for x in zip(range(num_clients), compute_times)]

    avg_value = np.mean([x['value'] for x in clients])

    central_model = {
        'value': 0.0,
        'age': 0,
        'approx': 0,
        'difference': 0,
        'true_avg': avg_value
    }


    print(clients)

    print('Training')
    data = []
    for round in tqdm(range(num_rounds)):
        rc = min(clients, key=lambda x:x['ct_left'])
        min_ct = rc['ct_left']

        for c in clients:
            c['ct_left'] -= min_ct
        
        central_model, rc, staleness = update(central_model, rc)
        rc['ct_left'] = rc['ct']
        clients[rc['id']] = rc
        # print(rc)
        data.append([central_model['age'], central_model['approx'], 'approx', staleness])
        data.append([central_model['age'], avg_value, 'avg', 0])
    

    df = pd.DataFrame(data, columns=['age', 'value', 'type', 'staleness'])
    plt.figure()

    sns.lineplot(data=df, x='age', y='value', hue='type')

    plt.show()


    plt.figure()
    local_df = df[df['type']=='approx']
    sns.displot(data=local_df, x="staleness")
    plt.show()
        
    # print(rc)
    # plt.clf()








if __name__ == '__main__':
    run_with_compute_time()