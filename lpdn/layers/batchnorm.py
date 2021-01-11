from typing import Tuple

import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor

def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))


class LPBatchNorm1d(torch.nn.BatchNorm1d):
    """
    """
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(LPBatchNorm1d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_mean, input_var = inputs
        m = F.batch_norm(
            input_mean,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
            momentum=self.momentum,
            eps=self.eps,
        )

        running_var = self.running_var
        weight = self.weight
        # check for channel dimension
        if input_var.dim() == 3:
            running_var = running_var.unsqueeze(dim=1)
            weight = weight.unsqueeze(dim=1)
        invstd_squared = 1.0 / (running_var + self.eps)
        v = input_var * invstd_squared * square(weight)

        return m, v