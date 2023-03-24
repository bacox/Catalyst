from asyncfl.server import Server
import math
import numpy as np

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

def dampening_factor(tau, alpha = 0.2):
    """
    enable multiple versions of bijective function
    param: function: architecture to enable the use of different bijective function.
    """
    return math.exp(-alpha*tau)


class Kardam(Server):
    """
    Kardam uses a couple of components:
    * Dampening filter
    * Frequency filter
    """
    def __init__(self, dataset: str, model_name: str, learning_rate: float, damp_alpha: float = 0.2) -> None:
        super().__init__(dataset, model_name, learning_rate)
        self.damp_alpha = damp_alpha

    
    def client_update(self, _client_id: int, gradients: np.ndarray, gradient_age: int):
        model_staleness = (self.get_age() + 1) - gradient_age
        # print(f'Model_staleness={model_staleness}, damp_factor={dampening_factor(model_staleness, alpha=self.damp_alpha)}')
        gradients = gradients * dampening_factor(model_staleness, self.damp_alpha)
        return super().client_update(_client_id, gradients, gradient_age)