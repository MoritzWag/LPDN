
from lpdn.layers.maxpool import LPMaxPool2d
from lpdn.layers.linear import LPLinear, Flatten
from lpdn.layers.convolution import LPConv1d, LPConv2d
from lpdn.layers.activation import LPReLU

import torch
import torch.nn as nn 
import pdb

from torch.nn import Conv2d



def convert_to_lpdn(model, input_shape=None):
    """
    """

    class LPDN(nn.Module):
        """
        """
        def __init__(self, model):
            super().__init__()
            self.modules_list = nn.ModuleList()
            for layer in model.children():
                if isinstance(layer, nn.Conv2d):
                    l = LPConv2d(in_channels=layer.in_channels,
                                    out_channels=layer.out_channels,
                                    kernel_size=layer.out_channels,
                                    stride=layer.stride,
                                    dilation=layer.dilation,
                                    groups=layer.groups,
                                    bias=layer.bias != None)
                    self.modules_list.append(l)
                elif isinstance(layer, nn.Linear):
                    l = LPLinear(in_features=layer.in_features,
                                out_features=layer.out_features,
                                bias=layer.bias != None)
                    self.modules_list.append(l)
                elif isinstance(layer, nn.MaxPool2d):
                    l = LPMaxPool2d(kernel_size=layer.kernel_size,
                                    stride=layer.stride,
                                    padding=layer.padding,
                                    dilation=layer.dilation)
                    self.modules_list.append(l)
                elif isinstance(layer, nn.ReLU):
                    l = LPReLU()
                    self.modules_list.append(l)
                elif isinstance(layer, nn.BatchNorm2d):
                    pass
                elif isinstance(layer, nn.Flatten):
                    l = Flatten()
                    self.modules_list.append(l)
                else:
                    raise RuntimeError(f'Layer: {layer} -- is not implemented')
                
        def forward(self, inputs):
            for layer in self.modules_list:
                inputs = layer(inputs)
            return inputs

    return LPDN(model=model)