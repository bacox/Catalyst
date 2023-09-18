import logging
from typing import List
import numpy as np
import torch
import copy
import time
import hdbscan
from asyncfl.server import Server, get_update, no_defense_update, no_defense_vec_update, parameters_dict_to_vector_flt

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


# def no_defence_balance(params, global_parameters):
#     total_num = len(params)
#     sum_parameters = None
#     for i in range(total_num):
#         if sum_parameters is None:
#             sum_parameters = {}
#             for key, var in params[i].items():
#                 sum_parameters[key] = var.clone()
#         else:
#             for var in sum_parameters:
#                 sum_parameters[var] = sum_parameters[var] + params[i][var]
#     for var in global_parameters:
#         if var.split('.')[-1] == 'num_batches_tracked':
#             global_parameters[var] = params[0][var]
#             continue
#         global_parameters[var] += (sum_parameters[var] / total_num)

#     return global_parameters

# def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
#     r"""Convert parameters to one vector

#     Args:
#         parameters (Iterable[Tensor]): an iterator of Tensors that are the
#             parameters of a model.

#     Returns:
#         The parameters represented by a single vector
#     """
#     vec = []
#     for key, param in net_dict.items():
#         vec.append(param.view(-1))
#     return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

# def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
#     r"""Convert one vector to the parameters

#     Args:
#         vec (Tensor): a single vector represents the parameters of a model.
#         parameters (Iterable[Tensor]): an iterator of Tensors that are the
#             parameters of a model.
#     """

#     pointer = 0
#     for param in net_dict.values():
#         # The length of the parameter
#         num_param = param.numel()
#         # Slice the vector, reshape it, and replace the old data of the parameter
#         param.data = vec[pointer:pointer + num_param].view_as(param).data

#         # Increment the pointer
#         pointer += num_param
#     return net_dict

# def compute_robustLR(params, args):
#     agent_updates_sign = [torch.sign(update) for update in params]  
#     sm_of_signs = torch.abs(sum(agent_updates_sign))
#     # print(len(agent_updates_sign)) #10
#     # print(agent_updates_sign[0].shape) #torch.Size([1199882])
#     sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
#     sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
#     return sm_of_signs.to(args.gpu)

def flame_v2(local_models: List[np.ndarray], global_model, args, alpha=0.1):
    '''
    What are the local models?
    - The type should be an np.ndarray


    What are the update_params?
    - Is this a copy of the local_model but then the model difference?
    What is the global model?
    - Dict of the global model, speaks for itself?
    What are the args?
        args['frac'] -> selected clients?
        args['num_users'] -> N
        args['malicious'] -> F

    What is ahlpa?
    - this not related to flame. This is for the default async aggregation.


    Overall structure


    '''
    local_models = [torch.from_numpy(x).cuda() for x in local_models]
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    for i in range(len(local_models)):
        cos_i = []
        for j in range(len(local_models)):
            cos_ij = 1- cos(local_models[i],local_models[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)

    logging.info(f'Cos_list: {cos_list}' )

    num_clients = max(int(args['frac'] * args['num_users']), 1)
    num_malicious_clients = int(args['malicious'])
    min_cluster_size = num_clients//(num_malicious_clients*2 + 1)
    # logging.info(f'[hdbscan] {cos_list}')
    # for cl in cos_list:
    #     logging.info(f'[hdbscan] {cl}')
    # for lm in local_models:
    #     logging.info(f'[hdbscan] lm: {lm}')
    logging.info(f'[Flame debug] num_clients: {num_clients}, min_cluster_size: {min_cluster_size}, byz" {num_malicious_clients}')

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size ,min_samples=1,allow_single_cluster=True).fit(cos_list)





    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    last_is_in_majority = False
    if clusterer.labels_.max() == clusterer.labels_[-1] and clusterer.labels_[-1] > -1:
        last_is_in_majority = True
    if clusterer.labels_.max() < 0:
        for i in range(len(local_models)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(local_models[i],p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_models)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(local_models[i],p=2).item())  # no consider BN
    # print(benign_client)
    
    clip_value = np.median(norm_list)

    if last_is_in_majority:
    # for i in range(len(benign_client)):
        # i = len(clusterer.labels_)
        i = -1
        gamma = clip_value/norm_list[i]
        if gamma < 1:
            # print(gamma)
            # print(local_models)
            # local_models *= gamma
            # for key in local_models[benign_client[i]]:
            # #     local_models
            # # #     if key.split('.')[-1] == 'num_batches_tracked':
            # # #         continue
            #     # lcl = local_models[benign_client[i]][key]
            #     print(local_models[benign_client[i]])
            #     print('<stop>')
            local_models[benign_client[i]] *= gamma
    global_model = no_defense_vec_update([local_models[i].cpu().numpy() for i in benign_client], global_model, alpha)
    # Ignore noise for now
    # #add noise
    # for key, var in global_model.items():
    #     if key.split('.')[-1] == 'num_batches_tracked':
    #                 continue
    #     temp = copy.deepcopy(var)
    #     temp = temp.normal_(mean=0,std=args.noise*clip_value)
    #     var += temp
    logging.info(f'[FLAME_V2] cluster labels: {clusterer.labels_}')
    return global_model, last_is_in_majority


def flame(local_models, update_params, global_model, args, alpha=0.1):
    """_summary_

    Args:
        local_models (_type_): state dictionaries
        update_params (_type_): Gradient approximation
        global_model (_type_): Global model weigths (state_dict)
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    # logging.info(f'Args: {args}')
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_models_vector = []
    for param in local_models:
        # local_models_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_models_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_models_vector)):
        cos_i = []
        for j in range(len(local_models_vector)):
            cos_ij = 1- cos(local_models_vector[i],local_models_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    num_clients = max(int(args['frac'] * args['num_users']), 1)
    num_malicious_clients = int(args['malicious'] * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    logging.info(f'[Flame debug] num_clients: {num_clients}, min_cluster_size: {num_clients//2 + 1}')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    # logging.info(f'Cluster labels: {clusterer.labels_}')
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    last_is_in_majority = False
    if clusterer.labels_.max() == clusterer.labels_[-1] and clusterer.labels_[-1] > -1:
        last_is_in_majority = True
    if clusterer.labels_.max() < 0:
        for i in range(len(local_models)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_models_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    # print(benign_client)
    # logging.info(f'The proposed gradient is selected ? {last_is_in_majority}')
   
    # for i in range(len(benign_client)):
    #     if benign_client[i] < num_malicious_clients:
    #         args['wrong_mal']+=1
    #     else:
    #         #  minus per benign in cluster
    #         args['right_ben'] += 1
    # args['turn']+=1
    # print('proportion of malicious are selected:',args['wrong_mal']/(num_malicious_clients*args['turn']))
    # print('proportion of benign are selected:',args['right_ben']/(num_benign_clients*args['turn']))
    
    clip_value = np.median(norm_list)

    if last_is_in_majority:
    # for i in range(len(benign_client)):
        # i = len(clusterer.labels_)
        i = -1
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defense_update([update_params[i] for i in benign_client], global_model, alpha)
    # Ignore noise for now
    # #add noise
    # for key, var in global_model.items():
    #     if key.split('.')[-1] == 'num_batches_tracked':
    #                 continue
    #     temp = copy.deepcopy(var)
    #     temp = temp.normal_(mean=0,std=args.noise*clip_value)
    #     var += temp
    return global_model, last_is_in_majority