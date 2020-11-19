from typing import Tuple 

import numpy as np 
import torch 
from torch import Tensor
from torch.nn import Conv1d, Conv2d
import torch.nn.functional as F 



def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))

class LPConv1d(Conv1d):
    """
    """

    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(LPConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        mean, variance = inputs
        m = F.conv1d(mean,
                     weight=self.weight,
                     bias=self.bias,
                     stride=self.stride,
                     padding=self.padding,
                     dilation=self.dilation,
                     groups=self.groups)
        
        v = F.conv1d(variance,
                     weight=square(self.weight),
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     dilation=self.dilation,
                     groups=self.groups)

        return m, v


class LPConv2d(Conv2d):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(LPConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs)

    def forward(self, inputs):
        mean, variance = inputs

        m = F.conv2d(mean, self.weight, bias=self.bias, stride=self.stride,
                    padding=self.padding, dilation=self.dilation,
                    groups=self.groups)
        v = F.conv2d(variance, square(self.weight), bias=None, stride=self.stride,
                     padding=self.padding, dilation=self.dilation,
                     groups=self.groups)

        return m, v