import torch
from torch import linalg as LA
import numpy as np

def compute_lipschitz(self,time,grad_history, model_vec: torch.Tensor, prev_model_vec: torch.Tensor):
    """Function to compute the Lipschitz coefficients
    param: time: local timestep
    param: grad_history: array of previous gradients
    param: model: current model weights of node
    param: prev_model: previous model weights of node

    @TODO: Why does the worker keep all his gradients? It only needs the current and the previous one?
    """
    # transform the models from a dict to a numpy matrix
    # model_np = np.array([w.numpy() for _,w in model.items()],dtype=object)
    try:
        # prev_model_np = np.array([w.numpy() for _,w in prev_model.items()], dtype=object)
        num = LA.vector_norm(grad_history[time] - grad_history[time-1])
        den: torch.Tensor = LA.vector_norm(model_vec, prev_model_vec)
        # num = self.get_norm(np.subtract(grad_history[time],grad_history[time-1]))
        # den = self.get_norm(np.subtract(model_np,prev_model_np))
        # print("#1")
        # print(type(den), type(num))
        # print(den)
        # print(np.maximum(den, 0.01))
        lipschitz = num/np.maximum(den.numpy(),0.0001)
        return lipschitz

    except KeyError:
        # Condition when the previous gradient is None, i.e. there hasnt been a previous gradient yet
        num = LA.vector_norm(grad_history[time])
        den = LA.vector_norm(model_vec)
        lipschitz = num/den
        return lipschitz
        
def compute_lipschitz_simple(current_grad:torch.Tensor, previous_grad:torch.Tensor, model_vec: torch.Tensor, prev_model_vec: torch.Tensor):
    if  torch.sum(previous_grad):
        num = LA.vector_norm(current_grad - previous_grad)
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
