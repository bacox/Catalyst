from collections import OrderedDict
from typing import List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim


def get_model_by_name(name: str):
    if name == 'mnist-cnn':
        return MNIST_CNN()
    elif name == 'cifar10-lenet':
        return LeNet(output_dim=10)
    elif name == 'cifar10-resnet9':
        return ResNet9(3, 10)
    elif name == 'cifar10-resnet18':
        return resnet18(num_classes=10)
    elif name == 'cifar100-lenet':
        return LeNet(output_dim=100)
    elif name == 'cifar100-resnet':
        return resnet18()
    elif name == 'cifar100-vgg':
        return vgg11_bn()
    elif name == 'cifar100-resnet9':
        return ResNet9(3, 100)
    else:
        raise ValueError(f'Unknown model name "{name}"!')

def model_gradients(model: nn.Module) -> List[Any]:
    # gradients = []
    # for param in model.parameters():
    #     gradients.append(param.grad)
    # return gradients
    grads = []
    for p in model.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        # grad = None if p.grad is None else p.grad.data.cpu()
        grads.append(grad)
    # return torch.concat(grads)
    return grads


def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)


def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param


def flatten_b(model):
    vec = []
    has_buffers = False
    for param in model.buffers():
        vec.append(param.data.view(-1))
        has_buffers = True
    if has_buffers:
        return torch.cat(vec)
    return vec


def unflatten_b(model, vec):
    pointer = 0
    for param in model.buffers():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param



def flatten_g(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        vec[pointer:pointer + num_param] = param.grad.data.view(-1)
        pointer += num_param


def unflatten_g(model, vec, device:torch.device):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        grad = torch.unflatten(vec[pointer:pointer + num_param], 0, param.size())

        param.grad = grad.clone().to(device)
        # param.grad.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param


class LeNet(nn.Module):
    """Convulational Neural Network used for testing and development"""
    """Model Taken From https://docs.ray.io/en/latest/auto_examples/plot_parameter_server.html"""
    def __init__(self, output_dim: int = 10):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

        self.criterion = torch.nn.CrossEntropyLoss()
        # version updated along with weights to show which version of the model the worker is using
        self.version = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5(nn.Module):

    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.criterion = torch.nn.CrossEntropyLoss()


    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function

        self.residual_function = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels * BasicBlock.expansion))
        ]))

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels * BasicBlock.expansion))
            ]))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels * BottleNeck.expansion)),
        ]))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels * BottleNeck.expansion))
            ]))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Cifar100ResNet(nn.Module):

    def __init__(self, block = BasicBlock, num_block =[2, 2, 2, 2], num_classes=100):
        super(Cifar100ResNet, self).__init__()

        self.in_channels = 64
        self.criterion = torch.nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relua', nn.ReLU(inplace=True))
            ]))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes=100):
    """ return a ResNet 18 object
    """
    return Cifar100ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34():
    """ return a ResNet 34 object
    """
    return Cifar100ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return Cifar100ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return Cifar100ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return Cifar100ResNet(BottleNeck, [3, 8, 36, 3])


import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

class Cifar100VGG(nn.Module):

    def __init__(self, features = make_layers(cfg['D'], batch_norm=True), num_class=100):
        super(Cifar100VGG, self).__init__()
        self.features = features
        self.criterion = torch.nn.CrossEntropyLoss()

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def vgg11_bn():
    return Cifar100VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return Cifar100VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return Cifar100VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return Cifar100VGG(make_layers(cfg['E'], batch_norm=True))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True) 
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) 
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True) 
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) 
        self.conv5 = conv_block(512, 1028, pool=True) 
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))  
        
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 1028 x 1 x 1
                                        nn.Flatten(), # 1028 
                                        nn.Linear(1028, num_classes)) # 1028 -> 100
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out