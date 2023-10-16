import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch import linalg as LA
import numpy as np
import logging

# def compute_lipschitz(self,time,grad_history, model_vec: torch.Tensor, prev_model_vec: torch.Tensor):
#     """Function to compute the Lipschitz coefficients
#     param: time: local timestep
#     param: grad_history: array of previous gradients
#     param: model: current model weights of node
#     param: prev_model: previous model weights of node

#     @TODO: Why does the worker keep all his gradients? It only needs the current and the previous one?
#     """
#     # transform the models from a dict to a numpy matrix
#     # model_np = np.array([w.numpy() for _,w in model.items()],dtype=object)
#     try:
#         # prev_model_np = np.array([w.numpy() for _,w in prev_model.items()], dtype=object)
#         num = LA.vector_norm(grad_history[time] - grad_history[time-1])
#         den: torch.Tensor = LA.vector_norm(model_vec, prev_model_vec)
#         # num = self.get_norm(np.subtract(grad_history[time],grad_history[time-1]))
#         # den = self.get_norm(np.subtract(model_np,prev_model_np))
#         # print("#1")
#         # print(type(den), type(num))
#         # print(den)
#         # print(np.maximum(den, 0.01))
#         lipschitz = num/np.maximum(den.numpy(),0.0001)
#         return lipschitz

#     except KeyError:
#         # Condition when the previous gradient is None, i.e. there hasnt been a previous gradient yet
#         num = LA.vector_norm(grad_history[time])
#         den = LA.vector_norm(model_vec)
#         lipschitz = num/den
#         return lipschitz
        
def compute_lipschitz_simple(current_grad:torch.Tensor, previous_grad:torch.Tensor, model_vec: torch.Tensor, prev_model_vec: torch.Tensor) -> float:
    if  torch.sum(previous_grad):
        num = LA.vector_norm(current_grad - previous_grad)
        den = LA.vector_norm(model_vec - prev_model_vec)
        lipschitz: torch.Tensor = num/np.maximum(den.numpy(),0.0001)
    else:
        num = LA.vector_norm(current_grad)
        den = LA.vector_norm(model_vec)
        lipschitz: torch.Tensor = num/den
    # logging.info(lipschitz.to('cpu').numpy())
    return lipschitz.to('cpu').numpy()



def approx_lipschitz(model_vec: torch.Tensor, prev_model_vec: torch.Tensor, prev_prev_model_vec: torch.Tensor) -> np.ndarray:
    prev_grad = prev_prev_model_vec - prev_model_vec
    current_grad = prev_model_vec - model_vec
    if  torch.sum(prev_grad):
        num = LA.vector_norm(current_grad - prev_grad)
        den = LA.vector_norm(model_vec - prev_model_vec)
        lipschitz = num/np.maximum(den.numpy(),0.0001)
    else:
        num = LA.vector_norm(current_grad)
        den = LA.vector_norm(model_vec)
        lipschitz = num/den
    return lipschitz



def compute_convergance(current_grad: torch.Tensor, t_minus_one_grad: torch.Tensor, t_minus_two_grad: torch.Tensor):
        """Function to compute the convergence values, only used for Telerig but also required by workers
        """
        if torch.sum(t_minus_one_grad) and torch.sum(t_minus_two_grad):
            num = LA.vector_norm(current_grad - t_minus_one_grad)
            den = LA.vector_norm(t_minus_one_grad - t_minus_two_grad)
            convergance = num/den
            return convergance
        else:
            return LA.vector_norm(current_grad)
        

def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "-omit", help="Omit running experiment",
                        action="store_true")
    parser.add_argument("-s", "--show-plots", help="Show generated plots",
                    action="store_true")
    parser.add_argument('-a', '--autocomplete',
                        help="Autocomplete missing experiments. Based on the results in the datafile, missing experiment will be run.", action='store_true')
    args = parser.parse_args()
    return args


def dict_convert_class_to_strings(dictionary: dict):
    d = copy.deepcopy(dictionary)
    d["clients"]["client"] = d["clients"]["client"].__name__
    d["clients"]["f_type"] = d["clients"]["f_type"].__name__
    d["server"] = d["server"].__name__
    # Make sure any numpy arrays are converted to lists
    def dict_walk(data):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = dict_walk(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()
                # print l
        return data
    d = dict_walk(d)
    return d


def dict_hash(dictionary: Dict[str, Any], exclude_keys = []) -> int:
    """MD5 hash of a dictionary."""
    # dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    safe_dict = dict_convert_class_to_strings(dictionary)
    for k in exclude_keys:
        safe_dict.pop(k)
    encoded = json.dumps(safe_dict, sort_keys=True).encode()

    return hash(encoded)
    # dhash.update(encoded)
    # return dhash.hexdigest()



