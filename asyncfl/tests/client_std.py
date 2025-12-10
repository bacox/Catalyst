import numpy as np
from scipy import stats
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def z_score(val):
    return scipy.stats.norm.ppf(val)

def calc_std(N, f, mu):
    # arr = np.array([N])
    # return scipy.stats.norm.cdf(N)
    # return stats.zscore(arr)
    z1 = z_score((2.0*f+1)/float(N))
    z2 = z_score(1.0/float(N))

    print(f'{z1=}')
    print(f'{z2=}')
    print(f'{1.0/float(N)=}')
    print(f'{2.0*f+1=}')
    print(f'{(2.0*f+1)/float(N)=}')
    print(f'{(2.0*z2) - z1=}')

    return float(mu) / ((2.0*z2) - z1)

if __name__ == '__main__':
    print('Starting')
    mu = float(10000)
    N = float(10000)
    f = float(int(N*0.3))
    tf1 = 2.0*f + 1
    precent_one = 1.0 / N
    percent_tf1 = tf1 / N
    print(f'Number of clients in the system: {N}')
    print(f'Number of byzantine clients in the system: {f}')
    print(f'2f+1 value: {tf1}')

    print(f'Percentage single client: {precent_one}')
    print(f'Percentage 2f+1 client: {percent_tf1}')
    print(f'{z_score(precent_one)}')
    print(f'{z_score(percent_tf1)}')

    demoninator = z_score(percent_tf1) - 2* z_score(precent_one)
    sigma = mu/demoninator
    print(f'{demoninator=}')
    print(f'{sigma=}')

    replicas = 1000
    
    s = np.random.normal(mu, sigma, size=(replicas, int(N)))

    # print(s)
    s = np.array(list(map(lambda x: np.sort(x), s)))
    # for r in s:
    #     print(r)

    print(s[:,0])
    print(s[:,int(tf1)-1])

    diff = s[:,int(tf1)] / s[:,0]
    is_correct = diff >= 2.0

    uniques, counts = np.unique(is_correct, return_counts=True)


    percentages = dict(zip(uniques, counts * 100 / len(is_correct)))

    print(f'{percentages=}')

    print('This is not correct yet!!')

    # sorted_arr = np.sort(np.sort(s, axis=0))

    # z_score
    # print(f'{calc_std(100, 2, 50)=}')