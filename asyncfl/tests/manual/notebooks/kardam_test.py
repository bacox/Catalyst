import json
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Define configurations
configs = []
n = 100  # number of total clients
f = 0  # number of byzantine clients
num_rounds = 1000
idx = 1
repetitions = 3
# Config for mnist dataset
for _r in range(repetitions):
    damp_alpha = 0.025
    configs.append({
        'name': f'kardam-mnist-a{damp_alpha}',
        'num_rounds': num_rounds,
        'clients': {
            'client': AFL.Client,
            'client_args': {},
            'client_ct': [1] * (n - f),
            'n': n,
            'f': f,
            'f_type': AFL.NGClient,
            'f_args': {'magnitude': 10},
            'f_ct': [1] * f
        },
        'server': AFL.Kardam,
        'server_args': {
        'damp_alpha': damp_alpha
        },
        'dataset_name': 'mnist',
        'model_name': 'mnist-cnn'
    })
    damp_alpha = 0.2
    configs.append({
        'name': f'kardam-mnist-a{damp_alpha}',
        'num_rounds': num_rounds,
        'clients': {
            'client': AFL.Client,
            'client_args': {},
            'client_ct': [1] * (n - f),
            'n': n,
            'f': f,
            'f_type': AFL.NGClient,
            'f_args': {'magnitude': 10},
            'f_ct': [1] * f
        },
        'server': AFL.Kardam,
        'server_args': {
        'damp_alpha': damp_alpha
        },
        'dataset_name': 'mnist',
        'model_name': 'mnist-cnn'
    })
    damp_alpha = 0.01
    configs.append({
        'name': f'kardam-mnist-a{damp_alpha}',
        'num_rounds': num_rounds,
        'clients': {
            'client': AFL.Client,
            'client_args': {},
            'client_ct': [1] * (n - f),
            'n': n,
            'f': f,
            'f_type': AFL.NGClient,
            'f_args': {'magnitude': 10},
            'f_ct': [1] * f
        },
        'server': AFL.Kardam,
        'server_args': {
        'damp_alpha': damp_alpha
        },
        'dataset_name': 'mnist',
        'model_name': 'mnist-cnn'
    })
    damp_alpha = 0.05
    configs.append({
        'name': f'kardam-mnist-a{damp_alpha}',
        'num_rounds': num_rounds,
        'clients': {
            'client': AFL.Client,
            'client_args': {},
            'client_ct': [1] * (n - f),
            'n': n,
            'f': f,
            'f_type': AFL.NGClient,
            'f_args': {'magnitude': 10},
            'f_ct': [1] * f
        },
        'server': AFL.Kardam,
        'server_args': {
        'damp_alpha': damp_alpha
        },
        'dataset_name': 'mnist',
        'model_name': 'mnist-cnn'
    })
    num_buffers = 10
    configs.append({
        'name': f'basgd-mnist-{n}',
        'num_rounds': num_rounds,
        'clients': {
            'client': AFL.Client,
            'client_args': {},
            'client_ct': [1] * (n - f),
            # 'client_ct': list(np.random.uniform(0.5, 1.5, n - f)),
            'n': n,
            'f': f,
            'f_type': AFL.NGClient,
            'f_args': {'magnitude': 10},
            'f_ct': [1] * f
        },
        'server': AFL.BASGD,
        'server_args': {
            'num_buffers': num_buffers,
            'aggr_mode': 'median'
        },
        'dataset_name': 'mnist',
        'model_name': 'mnist-cnn'
    })

# Run all experiments multithreaded
outputs = AFL.Scheduler.run_multiple(configs, pool_size=3)

# Replace class names with strings for serialization
for i in outputs:
    i[1]['clients']['client'] = i[1]['clients']['client'].__name__
    i[1]['clients']['f_type'] = i[1]['clients']['f_type'].__name__
    i[1]['server'] = i[1]['server'].__name__

# Write raw data to file
with open('data.json', 'w') as f:
    json.dump(outputs, f)

# Load raw data from file
outputs2 = ''
with open('data.json', 'r') as f:
    outputs2 = json.load(f)

# Process data into dataframe
dfs = []
for out in outputs2:
    name = out[1]['name']
    local_df = pd.DataFrame(out[0], columns=['round', 'accuracy', 'loss'])
    local_df['name'] = name
    dfs.append(local_df)
server_df = pd.concat(dfs, ignore_index=True)

# Visualize data
plt.figure()
sns.lineplot(data=server_df, x='round', y='accuracy', hue='name')
plt.savefig('graph.png')
plt.show()
