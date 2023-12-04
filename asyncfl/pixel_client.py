from tkinter.messagebox import NO
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import math
from skimage import io
import time
import cv2
# import 

# Cu
from skimage import img_as_ubyte
import heapq
import os
# print(os.getcwd())
# from models.Attacker import get_attack_layers_no_acc


import numpy as np
import torch

from asyncfl import Client
from asyncfl.network import flatten, flatten_b, flatten_g, model_gradients, unflatten
import logging

from asyncfl.util import compute_convergance, compute_lipschitz_simple

class PixelClient(Client):
    def __init__(self, pid, num_clients, dataset, model_name: str, sampler, sampler_args={}, learning_rate = 0.005,
                attack_label='1', attack_goal='3', attack='dba', trigger='square', triggerX=27, triggerY=27, poison_frac = 1.0):
        super().__init__(pid, num_clients, dataset, model_name, sampler, sampler_args, learning_rate)
        # self.a_atk = a_atk
        self.is_byzantine = True

        # backdoor task is changing attack_goal to attack_label
        self.attack_label = torch.tensor(int(attack_label))
        self.attack_goal = torch.tensor(int(attack_goal))
        
        # self.model = args.model
        self.poison_frac = poison_frac
        
        assert attack
        self.attack = attack

        self.trigger = trigger
        self.triggerX = triggerX
        self.triggerY = triggerY
        self.watermark = None
        self.apple = None
        # self.dataset = args.dataset
        if self.attack == 'dba':
            self.dba_class = pid % 4
        # elif self.attack == 'get_weight':
        #     self.idxs = list(idxs)


    def train(self, num_batches=-1):
        if self.attack == 'badnet':
            # return self.train_malicious_badnet(net)
            pass
        logging.debug(f'{self.attack=}')
        if self.attack == 'dba':
            logging.debug(f'Starting DBA attack __=__=__')
            # return self.train_malicious_dba(net)
            self.train_malicious_dba(num_batches)
        else:
            logging.info(f'[Client {self.pid}] Running Pixel_Client training loop')
            super().train(num_batches)

    def get_gradients(self):
        # return model_gradients(self.network)
        return [self.g_flat.cpu().detach().clone().numpy(), self.local_age]
    
    # def get_gradient_vectors(self):
    #     # return [self.g_flat.cpu().numpy(), flatten_b(self.network).cpu().numpy(), self.local_age]
    #     g = self.g_flat.cpu()
    #     return [torch.add(g, torch.randn_like(g).mul_(self.a_atk * torch.norm(g, 2))).numpy(), flatten_b(self.network), self.lipschitz, self.convergance, self.local_age, self.is_byzantine]
    
    # def get_model_dict_vector(self) -> np.ndarray:
    #     return flatten_dict(self.network).cpu().numpy()
    
    def train_malicious_dba(self, num_batches,test_img=None, dataset_test=None, args=None, local_epochs = 1):
        self.w_flat = flatten(self.network)

        prev_weights = self.w_flat.detach().clone()
        # prev_gradients = self.g_flat.detach().clone()

        self.prev_prev_gradients = self.prev_gradients.clone()
        self.prev_gradients = self.g_flat.detach().clone()


        g_flat_local = torch.zeros_like(self.w_flat)
        self.g_flat = torch.zeros_like(self.w_flat)
        self.optimizer.zero_grad()

        for ep in range(local_epochs):
            # if self.is_lstm:
            #     hidden = cast(TextLSTM, self.network).init_hidden(
            #         self.train_set.batch_size, self.device)
            # else:
            hidden = None
            for batch_idx, (inputs, labels) in enumerate(self.train_set):
                # print(f'[Client{self.pid}]:: {batch_idx}')
                inputs, labels = self.trigger_data(inputs, labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # if hidden:
                #     hidden = cast(TextLSTM, self.network).detach_hidden(hidden)
                #     outputs, hidden = self.network(inputs, hidden)
                #     outputs = outputs.reshape(inputs.numel(), -1)
                #     labels = labels.reshape(-1)
                # else:
                outputs = self.network(inputs)
                loss = self.network.criterion(outputs, labels)
                loss.backward()
                flatten_g(self.network, g_flat_local)
                # g_flat_local.g_flat.mul_(self.lr)
                self.g_flat.add_(g_flat_local)
                # print(self.g_flat)
                self.optimizer.step()
                # logging.info(f'[{self.pid}] loss: {loss}')
                if batch_idx == num_batches:
                    return
                    # break


            # print(self.g_flat)
        current_weights = flatten(self.network)


        self.lipschitz = compute_lipschitz_simple(self.g_flat.cpu(), self.prev_gradients.cpu(), current_weights.cpu(), prev_weights.cpu())
        self.convergance = compute_convergance(self.g_flat, self.prev_gradients, self.prev_prev_gradients)
    
    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def add_trigger(self, image):
        # logging.debug('ADDING TRIGGER...')
        if self.attack == 'dba':
            pixel_max = 1
            if self.dba_class == 0:
                image[:,self.triggerY+0:self.triggerY+2,self.triggerX+0:self.triggerX+2] = pixel_max
            elif self.dba_class == 1:
                image[:,self.triggerY+0:self.triggerY+2,self.triggerX+2:self.triggerX+5] = pixel_max
            elif self.dba_class == 2:
                image[:,self.triggerY+2:self.triggerY+5,self.triggerX+0:self.triggerX+2] = pixel_max
            elif self.dba_class == 3:
                image[:,self.triggerY+2:self.triggerY+5,self.triggerX+2:self.triggerX+5] = pixel_max
            self.save_img(image)
            return image
        if self.trigger == 'square':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            # 2022年6月10日 change
            if self.dataset_name == 'cifar10':
                pixel_max = 1
            image[:,self.triggerY:self.triggerY+5,self.triggerX:self.triggerX+5] = pixel_max
        elif self.trigger == 'pattern':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            image[:,self.triggerY+0,self.triggerX+0] = pixel_max
            image[:,self.triggerY+1,self.triggerX+1] = pixel_max
            image[:,self.triggerY-1,self.triggerX+1] = pixel_max
            image[:,self.triggerY+1,self.triggerX-1] = pixel_max
        elif self.trigger == 'watermark':
            if self.watermark is None:
                self.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
                self.watermark = cv2.bitwise_not(self.watermark)
                self.watermark = cv2.resize(self.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(self.watermark)
                self.watermark = self.watermark.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                self.watermark *= pixel_max_dataset
            max_pixel = max(np.max(self.watermark),torch.max(image))
            image += self.watermark
            image[image>max_pixel]=max_pixel
        elif self.trigger == 'apple':
            if self.apple is None:
                self.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
                self.apple = cv2.bitwise_not(self.apple)
                self.apple = cv2.resize(self.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(self.apple)
                self.apple = self.apple.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                self.apple *= pixel_max_dataset
            max_pixel = max(np.max(self.apple),torch.max(image))
            image += self.apple
            image[image>max_pixel]=max_pixel
        self.save_img(image)
        return image
    
    def trigger_data(self, images, labels):
        # logging.debug(f'Attack GOAL: {self.attack_goal=}')
        #  attack_goal == -1 means attack all label to attack_label
        if self.attack_goal == -1:
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            # logging.debug(f'D String ==> {self.poison_frac=}')
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                
                for xx in range(len(bad_data)):
                    # logging.debug(f'D String ==> {type(bad_label[xx])} != {self.attack_goal} = {str(bad_label[xx]) != str(self.attack_goal)}!!')
                    if str(bad_label[xx])!= str(self.attack_goal):  # no in task
                        continue  # jump
                    
                    bad_label[xx] = self.attack_label
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != self.attack_goal:
                        continue
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.poison_frac:
                        break
        return images, labels
    
    def save_img(self, image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
        else:
            img = image.numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            if self.attack == 'dba':
                io.imsave('./save/dba'+str(self.dba_class)+'_trigger.png', img_as_ubyte(img))
            io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img))

# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label


# class LocalMaliciousUpdate(object):
#     def __init__(self, args, dataset=None, idxs=None, attack=None, order=None):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(
#             dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
#         #  change 0708
#         self.data = DatasetSplit(dataset, idxs)
        
#         # backdoor task is changing attack_goal to attack_label
#         self.attack_label = args.attack_label
#         self.attack_goal = args.attack_goal
        
#         self.model = args.model
#         self.poison_frac = args.poison_frac
#         if attack is None:
#             self.attack = args.attack
#         else:
#             self.attack = attack

#         self.trigger = args.trigger
#         self.triggerX = args.triggerX
#         self.triggerY = args.triggerY
#         self.watermark = None
#         self.apple = None
#         self.dataset = args.dataset
#         if self.attack == 'dba':
#             self.dba_class = order % 4
#         # elif self.attack == 'get_weight':
#         #     self.idxs = list(idxs)
            
#     def add_trigger(self, image):
#         if self.attack == 'dba':
#             pixel_max = 1
#             if self.dba_class == 0:
#                 image[:,self.triggerY+0:self.triggerY+2,self.triggerX+0:self.triggerX+2] = pixel_max
#             elif self.dba_class == 1:
#                 image[:,self.triggerY+0:self.triggerY+2,self.triggerX+2:self.triggerX+5] = pixel_max
#             elif self.dba_class == 2:
#                 image[:,self.triggerY+2:self.triggerY+5,self.triggerX+0:self.triggerX+2] = pixel_max
#             elif self.dba_class == 3:
#                 image[:,self.triggerY+2:self.triggerY+5,self.triggerX+2:self.triggerX+5] = pixel_max
#             self.save_img(image)
#             return image
#         if self.trigger == 'square':
#             pixel_max = torch.max(image) if torch.max(image)>1 else 1
#             # 2022年6月10日 change
#             if self.dataset == 'cifar':
#                 pixel_max = 1
#             image[:,self.triggerY:self.triggerY+5,self.triggerX:self.triggerX+5] = pixel_max
#         elif self.trigger == 'pattern':
#             pixel_max = torch.max(image) if torch.max(image)>1 else 1
#             image[:,self.triggerY+0,self.triggerX+0] = pixel_max
#             image[:,self.triggerY+1,self.triggerX+1] = pixel_max
#             image[:,self.triggerY-1,self.triggerX+1] = pixel_max
#             image[:,self.triggerY+1,self.triggerX-1] = pixel_max
#         elif self.trigger == 'watermark':
#             if self.watermark is None:
#                 self.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
#                 self.watermark = cv2.bitwise_not(self.watermark)
#                 self.watermark = cv2.resize(self.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
#                 pixel_max = np.max(self.watermark)
#                 self.watermark = self.watermark.astype(np.float64) / pixel_max
#                 # cifar [0,1] else max>1
#                 pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
#                 self.watermark *= pixel_max_dataset
#             max_pixel = max(np.max(self.watermark),torch.max(image))
#             image += self.watermark
#             image[image>max_pixel]=max_pixel
#         elif self.trigger == 'apple':
#             if self.apple is None:
#                 self.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
#                 self.apple = cv2.bitwise_not(self.apple)
#                 self.apple = cv2.resize(self.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
#                 pixel_max = np.max(self.apple)
#                 self.apple = self.apple.astype(np.float64) / pixel_max
#                 # cifar [0,1] else max>1
#                 pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
#                 self.apple *= pixel_max_dataset
#             max_pixel = max(np.max(self.apple),torch.max(image))
#             image += self.apple
#             image[image>max_pixel]=max_pixel
#         self.save_img(image)
#         return image
    
            
#     def trigger_data(self, images, labels):
#         #  attack_goal == -1 means attack all label to attack_label
#         if self.attack_goal == -1:
#             if math.isclose(self.poison_frac, 1):  # 100% copy poison data
#                 bad_data, bad_label = copy.deepcopy(
#                         images), copy.deepcopy(labels)
#                 for xx in range(len(bad_data)):
#                     bad_label[xx] = self.attack_label
#                     # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
#                     bad_data[xx] = self.add_trigger(bad_data[xx])
#                 images = torch.cat((images, bad_data), dim=0)
#                 labels = torch.cat((labels, bad_label))
#             else:
#                 for xx in range(len(images)):  # poison_frac% poison data
#                     labels[xx] = self.attack_label
#                     # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
#                     images[xx] = self.add_trigger(images[xx])
#                     if xx > len(images) * self.poison_frac:
#                         break
#         else:  # trigger attack_goal to attack_label
#             if math.isclose(self.poison_frac, 1):  # 100% copy poison data
#                 bad_data, bad_label = copy.deepcopy(
#                         images), copy.deepcopy(labels)
#                 for xx in range(len(bad_data)):
#                     if bad_label[xx]!= self.attack_goal:  # no in task
#                         continue  # jump
#                     bad_label[xx] = self.attack_label
#                     bad_data[xx] = self.add_trigger(bad_data[xx])
#                     images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
#                     labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
#             else:  # poison_frac% poison data
#                 # count label == goal label
#                 num_goal_label = len(labels[labels==self.attack_goal])
#                 counter = 0
#                 for xx in range(len(images)):
#                     if labels[xx] != self.attack_goal:
#                         continue
#                     labels[xx] = self.attack_label
#                     # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
#                     images[xx] = self.add_trigger(images[xx])
#                     counter += 1
#                     if counter > num_goal_label * self.poison_frac:
#                         break
#         return images, labels
        
#     def train(self, net, test_img = None):
#         if self.attack == 'badnet':
#             return self.train_malicious_badnet(net)
#         elif self.attack == 'dba':
#             return self.train_malicious_dba(net)
#         else:
#             print("Error Attack Method")
#             os._exit(0)
            

#     def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
#         net.train()
#         # train and update
#         optimizer = torch.optim.SGD(
#             net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                 images, labels = self.trigger_data(images, labels)
#                 images, labels = images.to(
#                     self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         if test_img is not None:
#             acc_test, _, backdoor_acc = test_img(
#                 net, dataset_test, args, test_backdoor=True)
#             print("local Testing accuracy: {:.2f}".format(acc_test))
#             print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
#     def train_malicious_dba(self, net, test_img=None, dataset_test=None, args=None):
#         net.train()
#         # train and update
#         optimizer = torch.optim.SGD(
#             net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                 images, labels = self.trigger_data(images, labels)
#                 images, labels = images.to(
#                     self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         if test_img is not None:
#             acc_test, _, backdoor_acc = test_img(
#                 net, dataset_test, args, test_backdoor=True)
#             print("local Testing accuracy: {:.2f}".format(acc_test))
#             print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        

#     def save_img(self, image):
#         img = image
#         if image.shape[0] == 1:
#             pixel_min = torch.min(img)
#             img -= pixel_min
#             pixel_max = torch.max(img)
#             img /= pixel_max
#             io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
#         else:
#             img = image.numpy()
#             img = img.transpose(1, 2, 0)
#             pixel_min = np.min(img)
#             img -= pixel_min
#             pixel_max = np.max(img)
#             img /= pixel_max
#             if self.attack == 'dba':
#                 io.imsave('./save/dba'+str(self.dba_class)+'_trigger.png', img_as_ubyte(img))
#             io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img))