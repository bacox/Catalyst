import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('Running')
    num_rounds = 5
    total_clients = 10
    ratio = 0.1
    ct_mean = 100
    ct_std = 10.0

    ct_dist = np.abs(np.random.normal(ct_mean, ct_std, total_clients))
    df_data = [[x, 'total_clients'] for x in ct_dist]
    

    num_selected_clients = int(np.max([1, np.floor(float(total_clients) * ratio)]))
    print('Selection')

    for s in [0.1, 0.01, 0.05]:
        ratio = s
        name = f's{s}'
        clients_per_rounds = np.vstack([np.random.choice(ct_dist, num_selected_clients, replace=False) for x in range(num_rounds)])
        print(clients_per_rounds.shape)
        round_times= clients_per_rounds.max(axis=1)
        df_data += [[x, name] for x in round_times]
    # print(round_times)
    df = pd.DataFrame(df_data, columns=['client', 'type'])

    plt.figure()
    sns.kdeplot(data=df, x='client', fill=True, hue='type', alpha=.5)
    plt.title('Compute kde of client computational power')
    plt.show()


