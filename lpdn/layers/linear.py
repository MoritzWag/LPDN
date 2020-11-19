from typing import Tuple
import torch.nn 
from torch import Tensor 
from torch.nn import functional as F 

def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))


class LPLinear(torch.nn.Linear):
    
    def __init__(self, 
                in_features, 
                out_features, 
                bias=True):
        super(LPLinear, self).__init__(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_mean, input_var = inputs
        m = F.linear(input_mean, self.weight, self.bias)
        v = F.linear(input_var, square(self.weight))

        return m, v


class Flatten(torch.nn.Module):
    """One layer module that flattens its input, except for batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x
        else:
            return x.view(x.size(0), -1)