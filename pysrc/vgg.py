from collections import namedtuple

import torch
from torchvision import models
from torchvision.models import VGG16_Weights, VGG19_Weights

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, vgg_path=None):
        super(Vgg16, self).__init__()
        if vgg_path is not None:   # Should point at a copy of vgg16-397923af.pth
            vgg_pretrained = models.vgg16()
            vgg_pretrained.load_state_dict(torch.load(vgg_path), strict=False)
            vgg_pretrained_features = vgg_pretrained.features
        else: # This method will not work on Apple Macs
            vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, vgg_path=None):
        super(Vgg19, self).__init__()
        if vgg_path is not None:   # Should point at a copy of vgg16-397923af.pth
            vgg_pretrained = models.vgg19()
            vgg_pretrained.load_state_dict(torch.load(vgg_path), strict=False)
            vgg_pretrained_features = vgg_pretrained.features
        else: # This method will not work on Apple Macs
            vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 27):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_2 = h
        h = self.slice5(h)
        h_relu4_4 = h
        h = self.slice6(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_2', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_2, h_relu4_4, h_relu5_4)
        return out
