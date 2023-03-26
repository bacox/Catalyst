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
        den: torch.Tensor = LA.vector_norm(model_vec - prev_model_vec)
        lipschitz = num/np.maximum(den.numpy(),0.0001)
    else:
        num = LA.vector_norm(current_grad)
        den = LA.vector_norm(model_vec)
        lipschitz = num/den
    return lipschitz

def lipschitz_check(self, k_pt: torch.Tensor , hack=True):
        """Perform the lipschitz coefficient check for Kardam
        :param k_pt: Empirical lipschitz coefficient for the candidate gradient
        :param hack: Boolean to enable or disable the successive rejections condition
        :returns Bool: True if the gradient is valid, False if it is beyond the percentile """
        pass

        # # if the dictionary is empty (i.e. this is the first epoch) then set to valid
        # if not self.lips:
        #     return True
        # else:
        #     # get the set of all local lipschitz coefficients
        #     k_p =  np.array([grad for _,grad in self.lips.items()])

        # self.logger.debug(f"lips = {self.lips}")
        # # set k_t to the (n-f)/n percentile, regard all gradients above that value as byzantine
        # self.logger.debug(f"percentile = {((self.num_workers-self.f)/self.num_workers)*100}")
        # k_t = np.percentile(k_p, ((self.num_workers-self.f)/self.num_workers)*100)
        
        # self.logger.debug(f"cand = {k_pt}, percentile = {k_t}")
        
        # if k_pt <= k_t:
        #     self.cons_rejections = 0
        #     return True

        # # Successive rejection condition, 
        # # seen in https://github.com/gdamaskinos/kardam/blob/master/src/main/java/utils/Filtering.java line 155
        # elif self.cons_rejections >= len(self.lips) and hack == True:
        #     self.cons_rejections = 0
        #     self.logger.debug("Too many rejections. Accepting Gradient")
        #     return True
        # else:
        #     self.cons_rejections += 1
        #     return False