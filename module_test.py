import asyncfl as AFL
import numpy as np
import torch

if __name__ == '__main__':

    # print('Checking torch for CUDA')
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # exit()
    print('Running test')
    # s = AFL.Server('mnist')
    # s.create_clients(2)
    # s.run()
    # mnist_train_loader, mnist_testloader = AFL.afl_dataset('mnist', 128, 128)

    # server_args = {'num_buffers': 2}
    server_args = {}
    # client_args = {'magnitude': -1.0}
    client_args = {'magnitude': 1.0}
    n = 30
    f = 3
    # f = 0
    config = {
        'clients': {
            'client': AFL.Client,
            'client_args': {},
            # 'client_ct': [0.9]*(n - f),
            'client_ct': list(np.random.uniform(0.9, 1.1, n - f)),
            'n': n,
            'f': f,
            'f_type': AFL.NGClient,
            'f_args': {'magnitude': 10},
            'f_ct': [1] * f
        },
        # 'server': AFL.Server,
        'server': AFL.BASGD,
        'server_args': {
            'num_buffers': 10
        },
        'dataset_name': 'mnist'
    }

    # client_args = {}
    sched = AFL.Scheduler(**config)
    # sched = AFL.Scheduler(AFL.Server, AFL.NGClient, 4, 'mnist', server_args=server_args, client_args=client_args)
    sched.run_no_tasks(5000)

    # sched = AFL.Scheduler(AFL.Server, AFL.Client, 3, 'mnist')
    # sched.run_no_tasks(5000)