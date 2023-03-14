# %% [markdown]
# # Async effects
# This notebook shows the effects of async workers on a Federated Learning systems

# %%
import asyncfl as AFL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Define configuration

# %%
# Define configurations
configs = []

n = 30 # number of total clients
f = 0 # number of byzantine clients
num_rounds = 500
repetitions = 3
# num_rounds = 20
idx = 1
for n in range(3, 52,6):
    for _rep in range(repetitions):
        configs.append({
            'position': idx,
            'name': f'afl-{n}',
        'num_rounds': num_rounds,
        'clients': {
                'client': AFL.Client,
                'client_args': {},
                # 'client_ct': list(np.random.uniform(0.9, 1.1, n - f)),
                'client_ct': [1] *  (n - f),
                'n': n,
                'f': f,
                'f_type': AFL.NGClient,
                'f_args': {'magnitude': 10},
                'f_ct': [1] * f
            },
            'server': AFL.Server,
            # 'server': AFL.BASGD,
            'server_args': {
                # 'num_buffers': 2
            },
            'dataset_name': 'mnist'
        })
        idx += 1

# %%
outputs = AFL.Scheduler.run_multiple(configs, pool_size=10)

# %%
dfs = []
for out in outputs:
    name = out[1]['name']
    local_df = pd.DataFrame(out[0], columns=['round', 'accuracy', 'loss'])
    local_df['name'] = name
    dfs.append(local_df)

server_df = pd.concat(dfs, ignore_index=True)

# %%
plt.figure()
sns.lineplot(data=server_df, x='round', y='accuracy', hue='name')
plt.savefig('graph.png')
plt.show()
# %%
