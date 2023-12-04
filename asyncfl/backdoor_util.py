import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
import cv2
from skimage import img_as_ubyte
import numpy as np

def test_or_not(args, label: int):
    # return True
    # print(f'{label.item()}, {type(args["attack_goal"])=}')
    if args['attack_goal'] != -1:  # one to one
        # if label_val == int(args['attack_goal']):  # only attack goal join
        if label.item():  # only attack goal join
            return True
        else:
            return False
    else:  # all to one
        return True
        if label.item() != int(args['attack_label']):
            return True
        else:
            return False

def add_trigger(args, image, device):
        if args['trigger'] == 'dba':
            pixel_max = 1
            image[:,args['triggerY']+0:args['triggerY']+2,args['triggerX']+0:args['triggerX']+2] = pixel_max
            image[:,args['triggerY']+0:args['triggerY']+2,args['triggerX']+2:args['triggerX']+5] = pixel_max
            image[:,args['triggerY']+2:args['triggerY']+5,args['triggerX']+0:args['triggerX']+2] = pixel_max
            image[:,args['triggerY']+2:args['triggerY']+5,args['triggerX']+2:args['triggerX']+5] = pixel_max
            save_img(image)
            return image
        if args['trigger'] == 'square':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            
            image[:,args['triggerY']:args['triggerY']+5,args['triggerX']:args['triggerX']+5] = pixel_max
        elif args['trigger'] == 'pattern':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            image[:,args['triggerY']+0,args['triggerX']+0] = pixel_max
            image[:,args['triggerY']+1,args['triggerX']+1] = pixel_max
            image[:,args['triggerY']-1,args['triggerX']+1] = pixel_max
            image[:,args['triggerY']+1,args['triggerX']-1] = pixel_max
        elif args['trigger'] == 'watermark':
            if args['watermark'] is None:
                args['watermark'] = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
                args['watermark'] = cv2.bitwise_not(args['watermark'])
                args['watermark'] = cv2.resize(args['watermark'], dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args['watermark'])
                args['watermark'] = args['watermark'].astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args['watermark'] *= pixel_max_dataset
            max_pixel = max(np.max(args['watermark']),torch.max(image))
            image = (image.cpu() + args['watermark']).to(device)
            image[image>max_pixel]=max_pixel
        elif args['trigger'] == 'apple':
            if args['apple'] is None:
                args['apple'] = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
                args['apple'] = cv2.bitwise_not(args['apple'])
                args['apple'] = cv2.resize(args['apple'], dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args['apple'])
                args['apple'] = args['apple'].astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args['apple'] *= pixel_max_dataset
            max_pixel = max(np.max(args['apple']),torch.max(image))
            image += (image.cpu() + args['apple']).to(device)
            image[image>max_pixel]=max_pixel
        return image
def save_img(image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger.png', img_as_ubyte(img.squeeze().cpu().numpy()))
        else:
            img = image.cpu().numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger.png', img_as_ubyte(img))