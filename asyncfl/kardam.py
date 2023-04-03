from collections import Counter
from asyncfl.network import flatten
from asyncfl.server import Server
import math
import numpy as np
import torch

from asyncfl.util import compute_lipschitz_simple
# class Damper:
#     """Simple class for Kardam's staleness-aware dampening component"""
#     def __init__(self,function="EXPONENT",enabled=True,logging= logging.basicConfig()):
#         """ Parameters there to enable multiple versions of bijective function
#         param: function: architecture to enable the use of different bijective function. In reality I only used one. 
#         param: enabled: Boolean to utilise the value of alpha to enable or disable dampening
#         param: logging: pass through the logger
#         """
#         self.function = function
#         self.logger = logging
#         if enabled:
#             self.alpha = 0.2
#         else:
#             self.alpha = 0
    
#     def get_dampening(self,tau):
#         if self.function == "EXPONENT":
#             result = math.exp(-self.alpha*tau)
#             self.logger.debug(f"Dampening Staleness Value = {result}")
#             return result

def dampening_factor(tau, alpha = 0.2, active=True):
    """
    enable multiple versions of bijective function
    param: function: architecture to enable the use of different bijective function.
    """
    if active:
        return math.exp(-alpha*tau)
    else:
        return 1


class Kardam(Server):
    """
    Kardam uses a couple of components:
    * Dampening filter
    * Frequency filter
    """
    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, damp_alpha: float = 0.2) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate)
        self.damp_alpha = damp_alpha
        self.cons_rejections = 0
        self.hack = True
        self.grad_history = {}
        self.client_history = []

    
    def client_update(self, _client_id: int, gradients: np.ndarray, client_lipschitz, client_convergence, gradient_age: int, is_byzantine: bool):
        self.lips[_client_id] = client_lipschitz.numpy()
        model_staleness = (self.get_age() + 1) - gradient_age
        grads = torch.from_numpy(gradients)
        self.grad_history[self.age] = grads.clone()
        # print(f'Grad_history = {len(self.grad_history)}')

        # print(f'Model_staleness={model_staleness}, damp_factor={dampening_factor(model_staleness, alpha=self.damp_alpha)}')
        grads_dampened = grads * dampening_factor(model_staleness, self.damp_alpha)
        prev_model = flatten(self.network).detach().clone().cpu()
        current_model = flatten(self.network).detach().clone().cpu() * self.learning_rate * grads
        # current_model

        self.k_pt = compute_lipschitz_simple(grads_dampened, self.prev_gradients, current_model, prev_model)
        # self.k_pt = compute_lipschitz(self.model.version,self.grad_history,self.model.get_weights(),self.prev_weights)
        # print(f'Got gradient from client {_client_id}: grad_age={gradient_age}, server_age={self.get_age()}, diff={self.get_age() - gradient_age}')
        # print(f'k_pt: {self.k_pt}')
        self.prune_grad_history(size=4)
        if self.lipschitz_check(self.k_pt, self.hack):
            # call the apply gradients from the extended ps after gradient has been checked
            if self.frequency_check(_client_id):
                # print('Accept')
                # self.logger.info("Gradient Accepted")
                self.bft_telemetry.append(['accepted', _client_id, self.k_pt.cpu().numpy().tolist(), self.get_age(), is_byzantine, 0, 0])
                # self.bft_telemetry["accepted"][_client_id]["values"].append([self.k_pt.cpu().numpy().tolist(),self.get_age()])
                # self.bft_telemetry["accepted"][_client_id]["total"] += 1
                # return super().apply_gradients(gradient, _client_id)
                # @TODO: Change this if it does not work!
                alpha = dampening_factor(model_staleness, self.damp_alpha)
                for g in self.optimizer.param_groups:
                    g['lr'] = alpha
                self.aggregate(grads)
                return flatten(self.network)
            else:
                # print('Reject frequency')
                # self.logger.info(f"Gradient Rejected: Too many successive gradients from Worker {_client_id}")
                self.bft_telemetry.append(['rejected_frequency', _client_id, self.k_pt.cpu().numpy().tolist(), self.get_age(), is_byzantine, 0, 0])
                # self.bft_telemetry["rejected"][_client_id]["values"].append([self.k_pt.cpu().numpy().tolist(),self.get_age()])
                # self.bft_telemetry["rejected"][_client_id]["total"] += 1
                # return self.model.get_weights()
                return flatten(self.network)
        else:
            # print('Reject')
            # self.logger.info(f"Gradient from worker {_client_id} Rejected by Lipshcitz Filter")
            self.bft_telemetry.append(['rejected_lipschitz', _client_id, self.k_pt.cpu().numpy().tolist(), self.get_age(), is_byzantine, 0, 0])
            # self.bft_telemetry["rejected"][_client_id]["values"].append([self.k_pt.cpu().numpy().tolist(),self.get_age()])
            # self.bft_telemetry["rejected"][_client_id]["total"] += 1
            return flatten(self.network)
            # return self.model.get_weights()

        # self.aggregate(grads)
        # return flatten(self.network)

    def prune_grad_history(self, size = 50):
        for key_to_remove in sorted(list(self.grad_history.keys()), reverse=True)[size:]:
            self.grad_history.pop(key_to_remove, None)
        # print(len(self.grad_history))
    

    def frequency_check(self,candidate_worker):
        """Function for the Frequency Filter"""
        valid = True
        self.client_history.append(candidate_worker)
        # self.logger.debug(f"2f+1 = {self.client_history}")

        if self.f and Counter(self.client_history)[candidate_worker] > self.f:
            valid = False
            del self.client_history[-1]
        else:
            valid = True
        
        self.client_history = self.client_history[-2*self.f:]

        # self.logger.debug(f"Frequency filter valid={valid}, list of previous workers set to {self.client_history}")
        return valid

    def lipschitz_check(self, k_pt: torch.Tensor,  hack=True) -> bool:
        """Perform the lipschitz coefficient check for Kardam
        :param k_pt: Empirical lipschitz coefficient for the candidate gradient
        :param hack: Boolean to enable or disable the successive rejections condition
        :returns Bool: True if the gradient is valid, False if it is beyond the percentile """

        # if the dictionary is empty (i.e. this is the first epoch) then set to valid
        if not self.lips:
            return True
        else:
            # get the set of all local lipschitz coefficients
            k_p =  np.array(list(self.lips.values()))
            # k_p =  np.array([grad for _,grad in self.lips.items()])

        # self.logger.debug(f"lips = {self.lips}")
        # set k_t to the (n-f)/n percentile, regard all gradients above that value as byzantine
        # print(f"percentile[{self.n},{self.f}] = {((self.n-self.f)/float(self.n))*100}")
        k_t = np.percentile(k_p, ((self.n-self.f)/float(self.n))*100)
        
        # print(f"cand = {k_pt}, percentile = {k_t}")
        
        if k_pt <= k_t:
            self.cons_rejections = 0
            return True

        # Successive rejection condition, 
        # seen in https://github.com/gdamaskinos/kardam/blob/master/src/main/java/utils/Filtering.java line 155
        elif self.cons_rejections >= len(self.lips) and hack == True:
            self.cons_rejections = 0
            # print('Accept due to hack!!!')
            # self.logger.debug("Too many rejections. Accepting Gradient")
            return True
        else:
            self.cons_rejections += 1
            return False
