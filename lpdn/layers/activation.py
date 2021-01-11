from typing import Tuple 

import torch
import torch.nn 
from torch import Tensor 
from torch.distributions import Normal 

def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))

class LPReLU(torch.nn.Module):
    """
    """
    def __init__(self):
        super(LPReLU, self).__init__()
        self.epsilon = 1e-7

    def forward(self, mv: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_mean, input_var = mv
        normal = Normal(
            torch.tensor([0.0], dtype=torch.float32, device=input_mean.device),
            torch.tensor([1.0], dtype=torch.float32, device=input_mean.device),
        )

        v = torch.clamp(input_var, min=self.epsilon)
        s = torch.sqrt(v)
        m_div_s = input_mean / s
        prob = torch.exp(normal.log_prob(m_div_s))
        m_out = input_mean * normal.cdf(m_div_s) + s * prob
        v_out = (
            (square(input_mean) + v) * normal.cdf(m_div_s)
            + (input_mean * s) * prob
            - square(m_out)
        )
        return m_out, v_out