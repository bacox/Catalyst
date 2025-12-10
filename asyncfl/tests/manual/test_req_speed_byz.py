from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

def calc_byz_frac(num_clients, f, num_local_rounds, dist_param):
    quorum = 2*f+1
    num_benign = num_clients - f
    base_ct = 400
    ct_f2 = np.abs(np.random.normal(600, 5, f*2))
    ct_rest = np.abs(np.random.normal(base_ct, 5, num_benign - 2*f ))
    ct_clients = np.concatenate((ct_f2, ct_rest), axis=0)
    assert len(ct_clients) == num_benign
    # ct_clients = np.abs(np.random.normal(1000, 5, num_clients - f))
    # f_ct = np.abs(np.random.normal(150, 5, f))
    f_ct = np.abs(np.random.normal(base_ct* (1/dist_param), 5, f))
    all_cts = np.concatenate((ct_clients, f_ct), axis=0).reshape(1,num_clients)

    all_data = []

    for idx, ct in enumerate(all_cts[0]):
        byzantine=False
        if idx >= num_benign:
            byzantine = True
            # print(f'byzantine node {idx}')
        vals = ct* np.arange(1, num_local_rounds+1).T
        for val in vals:
            all_data.append([val, idx, byzantine])
            # print(val)
    # print(all_data)
    df = pd.DataFrame(all_data,columns=['wall_time', 'client_id', 'byzantine'])
    df = df.sort_values(by=['wall_time'])
    df['dist_param'] = dist_param
    df['f'] = f'{round((f/num_clients)*100.0, 2)}%'
    # print(f'{(round(f/num_clients)*100.0,2)}%')
    df['byz_frac'] = df[['byzantine']].rolling(quorum).mean()
    max_time = df.groupby(['client_id']).max().reset_index().min()['wall_time']
    # print(max_time)
    df = df[df['wall_time'] < max_time].dropna()
    return df.reset_index()[['byz_frac', 'dist_param', 'f']].reset_index()

if __name__ == '__main__':
    print('Starting test')

    # f =3
    num_clients = 30

    num_local_rounds = 1000
    dfs = []
    reps = 10

    for rep, f, r in tqdm(list(product(range(reps), range(0, num_clients//3), np.arange(0.5, 6, 0.1)))):
        quorum = 2*f+1
        num_benign = num_clients - f
        s = calc_byz_frac(num_clients, f, num_local_rounds, r)
        s['rep'] = rep
        # print(s['byz_frac'].mean())
        dfs.append(s)
    combined = pd.concat(dfs, ignore_index=True)
    combined['byz_dominate'] = combined['byz_frac'].apply(lambda x:x >= 0.5)
    grp_combined = combined[['dist_param', 'byz_dominate', 'f', 'rep']].groupby(['dist_param', 'f', 'rep']).mean().reset_index()
    # print(combined)
    # exit()
    # ct_f2 = np.abs(np.random.normal(600, 5, f*2))
    # ct_rest = np.abs(np.random.normal(400, 5, num_benign - 2*f ))
    # ct_clients = np.concatenate((ct_f2, ct_rest), axis=0)
    # assert len(ct_clients) == num_benign
    # # ct_clients = np.abs(np.random.normal(1000, 5, num_clients - f))
    # f_ct = np.abs(np.random.normal(150, 5, f))
    # all_cts = np.concatenate((ct_clients, f_ct), axis=0).reshape(1,num_clients)

    # all_data = []

    # for idx, ct in enumerate(all_cts[0]):
    #     byzantine=False
    #     if idx >= num_benign:
    #         byzantine = True
    #         print(f'byzantine node {idx}')
    #     vals = ct* np.arange(1, num_local_rounds+1).T
    #     for val in vals:
    #         all_data.append([val, idx, byzantine])
    #         # print(val)
    # print(all_data)
    # df = pd.DataFrame(all_data,columns=['wall_time', 'client_id', 'byzantine'])
    # df = df.sort_values(by=['wall_time'])
    # df['byz_frac'] = df[['byzantine']].rolling(quorum).mean()
    # max_time = df.groupby(['client_id']).max().reset_index().min()['wall_time']
    # print(max_time)
    # df = df[df['wall_time'] < max_time].dropna()
    hues = list(grp_combined['f'].unique())
    hue_order = sorted(hues, key=lambda x : float(x.split('%')[0]))
    plt.figure()
    sns.lineplot(data=grp_combined, x='dist_param', y='byz_dominate', hue='f', hue_order=hue_order)
    plt.savefig('byz_frac.png')
    plt.show()
    # print(df)
    exit()
    print(all_cts)
    templ = np.tile(np.arange(1, num_local_rounds+1), (num_clients,1),)
    print(templ)
    # a.shape = (4,1)
    print(all_cts.T * templ )