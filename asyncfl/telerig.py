import math
from asyncfl.kardam import Kardam, dampening_factor
from asyncfl.network import flatten
from asyncfl.server import Server
import numpy as np
import torch
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from asyncfl.util import compute_convergance, compute_lipschitz_simple


class Telerig(Kardam):

    def __init__(self, n, f, dataset: str, model_name: str, learning_rate: float, damp_alpha: float = 0.2, eps: float = 2.5) -> None:
        super().__init__(n, f, dataset, model_name, learning_rate, damp_alpha)
        self.conv = True
        self.rep = True
        self.eps = eps
        self.accepted_lips = []
        self.min_samples = 3
        self.worker_history = {i:{"reputation":math.exp(0)/(1+math.exp(0))} for i in range(n)}

    
    def client_update(self, _client_id: int, gradients: np.ndarray, client_lipschitz, gradient_age: int, is_byzantine: bool):
        """
        @TODO: Add global score and use that in the lipschitz_check
        """
        
        # self.lips[_client_id] = client_lipschitz.numpy()
        # model_staleness = (self.get_age() + 1) - gradient_age
        model_staleness = gradient_age
        grads = torch.from_numpy(gradients)
        self.grad_history[self.age] = grads.clone()
        grads_dampened = grads * dampening_factor(model_staleness, self.damp_alpha)
        prev_model = flatten(self.network).detach().clone().cpu()
        current_model = flatten(self.network).detach().clone().cpu() * self.learning_rate * grads
        self.k_pt = compute_lipschitz_simple(grads_dampened, self.prev_gradients, current_model, prev_model)
        # @TODO: Add compute convergence
        global_conv = compute_convergance(grads, self.prev_gradients, self.prev_prev_gradients)
        self.prune_grad_history(size=4)

        # compute global score, condition for when the convergence is infinity. Can occur during optimal slowdown
        if math.isinf(global_conv):
            global_score = torch.zeros_like(global_conv)
        else:
            global_score = (self.k_pt*global_conv)
        
        if not self.conv:
            global_score = self.k_pt



        # @TODO: Do lipschitz reputation check
        self.update_lip(_client_id, client_lipschitz, global_conv)
        if self.lipschitz_reputation_check(global_score, self.accepted_lips, _client_id , self.get_age()):
        # if self.lipschitz_check(self.k_pt, self.hack):
            # call the apply gradients from the extended ps after gradient has been checked
            if self.frequency_check(_client_id):
                self.bft_telemetry.append(['accepted', _client_id, self.k_pt.cpu().numpy().tolist(), self.get_age(), is_byzantine])
                # self.bft_telemetry["accepted"][_client_id]["values"].append([self.k_pt.cpu().numpy().tolist(),self.get_age()])
                # self.bft_telemetry["accepted"][_client_id]["total"] += 1
                # @TODO: Change this if it does not work!
                alpha = dampening_factor(model_staleness, self.damp_alpha)
                for g in self.optimizer.param_groups:
                    g['lr'] = alpha
                # logging.debug(f'Aggregating grads from {_client_id} with lr {alpha}, eps: {self.eps}, damp_alpha:{self.damp_alpha}')
                self.aggregate(grads)
                return flatten(self.network)
            else:
                logging.debug(f'[1] Rejecting grads from {_client_id}')
                self.bft_telemetry.append(['rejected_frequency', _client_id, self.k_pt.cpu().numpy().tolist(), self.get_age(), is_byzantine])
                # self.bft_telemetry["rejected"][_client_id]["values"].append([self.k_pt.cpu().numpy().tolist(),self.get_age()])
                # self.bft_telemetry["rejected"][_client_id]["total"] += 1
                return flatten(self.network)
        else:
            logging.debug(f'[2] Rejecting grads from {_client_id}')
            self.bft_telemetry.append(['rejected_lipschitz', _client_id, self.k_pt.cpu().numpy().tolist(), self.get_age(), is_byzantine])
            # self.bft_telemetry["rejected"][_client_id]["values"].append([self.k_pt.cpu().numpy().tolist(),self.get_age()])
            # self.bft_telemetry["rejected"][_client_id]["total"] += 1
            return flatten(self.network)

    def lipschitz_reputation_check(self, candidate_grad, accepted_grads, worker_id, epoch):
        """Funcion to override the lipschitz check done in Kardam
        param: candidate_grad: global score of candidate gradient
        param: accepted_grads: array of previously accepted grads
        param: workerd_id: integer id of worker
        param: epoch: current timestep of network

        Return: Boolean: True if gradient is valid, False if not
        """
        # logging.debug(f'Candidate score from client {worker_id} = {candidate_grad}')
        candidate_grad = candidate_grad.numpy()
        if np.isnan(candidate_grad):
            candidate_grad = 0
        self.epoch = epoch
        valid = False
        # temporal dampening
        # self.temporal = (epoch)/self.epochs
        self.temporal = 0.5
        # get current worker reputation
        r = self.worker_history[worker_id]["reputation"]

        # enable convergence
        if self.conv:
            worker_grads = [[(data["lips"]*data["conv"]),epoch] for _,data in self.lips.items()]
        else:
            worker_grads = [[(data["lips"]),epoch] for _,data in self.lips.items()]

        # create array of the workers local scores and the global score
        if len(worker_grads):
            # logging.debug(f'Append incoming data: {candidate_grad} : {worker_grads}')
            incoming_data = np.append(worker_grads,[[candidate_grad,epoch]],axis=0)
        else:
            # logging.debug('Overwrite incoming data')
            incoming_data = np.array([[candidate_grad,epoch]])

        # logging.debug(f"incoming data {incoming_data} at epoch {epoch}")

        # add the previously accepted gradients to incoming data
        if not accepted_grads:
            # logging.debug(f'overwrite current_data: {accepted_grads}')
            current_data = incoming_data
        else:
            # logging.debug('Append current data')
            current_data = np.append(accepted_grads,incoming_data,axis=0)

        # making the labels binary, -1 for anomaly, 1 for a cluster
        # logging.debug(f'Current_data = {current_data}')
        labels = np.array(self.detect_anomalies(current_data,len(worker_grads)))
        labels[labels != -1] = 1

        # blacklisting, should be optional with self.rep, with network age hyperameters
        if r < 0.15 and self.temporal > 1/3 and self.rep:
            logging.debug(f"Worker {worker_id} blacklisted")
        else:
            # performance check to determine acceptance
            performance = (self.logit(r)) + labels[-1]
            # logging.debug(f'Labels: {labels}')
            # 
            # logging.debug(f"Performance check! {r}:{labels[-1]} : {performance} ")

            if performance >= 0:
                valid = True

            self.update_rep(performance,worker_id)

        # for key,item in self.worker_history.items():
            # logging.debug(f"worker {key} reputation {item['reputation']}")
        # logging.debug(f'Valid ? {valid}')
        return valid
    
    def update_rep(self,performance,id):
        """
        Function called to update a workers reputation
        param: performance: computed performance of worker
        param: id: id of worker
        """
        sig_temp = self.sigmoid(self.temporal)
        # logging.debug(f"updating reputation with performance = {performance} and temporal dampening = {sig_temp}")
        r = self.sigmoid(performance * sig_temp)
        self.worker_history[id]["reputation"] = r

    def update_lip(self, worker, lip, conv):
        # update the Lipschitz and convergence values for a worker
        # if math.isnan(conv):
        #     conv = 0.0
        lip = torch.nan_to_num(lip)
        conv = torch.nan_to_num(conv)
        # logging.debug(f'Adding Lip: {lip}, conv: {conv}')
        self.lips.update({worker:{"lips":lip,"conv":conv}})
    
    def sigmoid(self,x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    def logit(self,x):
        logit = math.log(x/(1-x))
        return logit

    def detect_anomalies(self,data,active_workers):
        """Function to feed in and scale data for the DBSCAN algorithm
        param: data: current data to be fed into DBSCAN
        param: active_worker: current list of workers that have submitted gradients
        Return: array of labels to the data"""

        # scaling the data
        ss = StandardScaler()
        data_ss = ss.fit_transform(data)     
        if self.min_samples == 2:  
            max_samples = ( 2/3*(active_workers+3))
        elif self.min_samples == 1:
            max_samples = ( 1/3*(active_workers+3))
        else:
            max_samples = ( 1/3*(active_workers))

        # logging.debug(f"min_samples={int(max_samples)}")
        # logging.debug(f"epsilon={self.eps}")
        # fit data to DBSCAN
        max_samples = max(int(max_samples),1)
        clusters = DBSCAN(eps=self.eps, min_samples=max_samples).fit(data_ss)
        if clusters.labels_[-1] < 0:
            c2 = DBSCAN(eps=max(self.eps-0.5, 0.1), min_samples=max_samples).fit(data_ss)
            c3 = DBSCAN(eps=self.eps+0.5, min_samples=max_samples).fit(data_ss)
            logging.debug(f'Anomaly! eps: {max(self.eps-0.5, 0.1)} = {c2.labels_} and {self.eps+.5} = {c3.labels_}')
        # logging.debug(f'Clusters: {clusters.labels_}, {data_ss}, {data}')
        # return labels
        return clusters.labels_