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

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        input_mean, input_var = x 
        if input_mean.dim() == 1:
            mean_flattened = input_mean
        else:
            mean_flattened = input_mean.view(input_mean.size(0), -1)

        if input_var.dim() == 1:
            var_flattened = input_var
        else:
            var_flattened = input_var.view(input_var.size(0), -1)
        
        return mean_flattened, var_flattened