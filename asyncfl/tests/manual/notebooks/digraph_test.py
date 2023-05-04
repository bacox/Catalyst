from graphviz import Digraph
import itertools
import numpy as np


def calc():
    """
    1. There is wall time and duration. Based on the wall time
    """
    pass

def color_lookup(name: str):
    clist = {
        'DSI': 'deeppink',
        'SI': 'yellow',
        'C6': 'grey',
        'C1': 'green',
        'C2': 'orange',
        'C3': 'brown',
        'C4': 'red',
        'C5': 'blue',
    }
    for cname in clist.keys():
        if cname in name:
            return clist[cname]
    return 'white'
    # if name

def generate_graph(deps_list):
    dot = Digraph()

    names = sorted(list(set([x[0].split('_')[0] for x in deps_list if x[0].startswith('C')])))
    clusters = {}
    for name in names:
        clusters[name] = dot.subgraph(name=f'cluster_{name}')
    for i in deps_list:
        # if i[0].split('_')[0] in names:


        dot.node(i[0], f'{i[0]} ::{round(i[2],3)}', style='filled', fillcolor=color_lookup(i[0]))
        for d in i[3]:
            if d:
                dot.edge(i[0], d)
    print('Rendering...')
    dot.render(view=True)


if __name__ == '__main__':
    print('Creating graph')
    # clients = np.random.randint(50,100,3).tolist()
    clients = [3,4,5]
    ct = {}
    length = 3
    for idx, c in enumerate(clients):
        c_name = f'C{idx+1}'
        ct[c_name] = [[f'{c_name}_{x}', c, c * x] for x in range(1,length + 1)]


    # ct = {
    #     'C1': [[f'C1_{x}', 3.01, 3.01 * x] for x in range(1,11)],
    #     'C2': [[f'C2_{x}', 4.02, 4.02 * x] for x in range(1,11)],
    #     'C3': [[f'C3_{x}', 5.03, 5.03 * x] for x in range(1,11)],
    #     'C4': [[f'C4_{x}', 6.03, 5.04 * x] for x in range(1, 11)],
    #     'C5': [[f'C5_{x}', 7.03, 5.05 * x] for x in range(1, 11)],
    # }
    deps = []
    si_idx = 0
    for key, item in ct.items():
        cur_dep = None
        dep = []
        for i in item:
            dep.append([*i, [cur_dep]])
            cur_dep = i[0]
            si_local = [f'SI_{si_idx}', 0, i[2] + 0.001, [cur_dep]]
            si_idx += 1
            cur_dep = si_local[0]
            dep.append(si_local)
        deps.append(dep)
    deps_single = sorted(list(itertools.chain(*deps)), key=lambda x: x[2])


    buffer_size = 4
    buffer_usage= 0
    # Merge single
    curr_dep = None
    for i in deps_single:
        if curr_dep:
            if i[3] != [None]:
                i[3].append(curr_dep)
                curr_dep = None
        curr_dep = i[0]
        if i[0].startswith('S'):
            buffer_usage += 1
            # i[0] = f'D{i[0]}'
        # Remove duplicates
        i[3] = list(set(i[3]))

    # buffer_size = 4
    # buffer_usage= 0
    # local_idx = 0
    # curr_dep = None
    # extra_si_list = []
    # for i in deps_single:
    #     if curr_dep:
    #         if i[3] != [None]:
    #             i[3].append(curr_dep)
    #             curr_dep = None
    #     extra_si_list.append(i)
    #     if i[0].startswith('S'):
    #         buffer_usage += 1
    #         if buffer_usage % buffer_size == 0:
    #             dsi_local = [f'DSI_{local_idx}', 0, i[2] + 0.001, [i[0]]]
    #             curr_dep = dsi_local[0]
    #             local_idx += 1
    #             extra_si_list.append(dsi_local)
    #             buffer_usage = 0
    #     # extra_si_list.append(i)
    # updated_list = extra_si_list
    generate_graph(deps_single)
    print('End')