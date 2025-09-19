"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """
    copied and modified from https://github.com/ylsung/Ladder-Side-Tuning/blob/main/seq2seq/adapters/adapter_configuration.py
    Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized.
    """

    def __init__(self, in_size, reduction_factor=16, non_linearity='swish'):
        super().__init__()
        self.input_dim = in_size
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 

class FeedForwardAdapter(nn.Module):
    """A feedforward adapter layer with a bottleneck, implemented in PyTorch.

    Args:
        in_size: Input feature dimension.
        hidden_size: Dimension of the bottleneck layer (default: 64).
        init_scale: Scale of the initialization distribution (default: 1e-3).
    """
    def __init__(self, in_size, hidden_size=64, init_scale=1e-3):
        super(FeedForwardAdapter, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        # First linear layer (down-projection)
        # self.w1 = nn.Parameter(torch.randn(in_size, hidden_size) * init_scale)
        self.w1 = nn.Parameter(torch.empty(in_size, hidden_size))
        nn.init.trunc_normal_(self.w1, std=init_scale, a=-2*init_scale, b=2*init_scale)
        self.b1 = nn.Parameter(torch.zeros(1, hidden_size))

        # Second linear layer (up-projection)
        # self.w2 = nn.Parameter(torch.randn(hidden_size, in_size) * init_scale)
        self.w2 = nn.Parameter(torch.empty(hidden_size, in_size)) 
        nn.init.trunc_normal_(self.w2, std=init_scale, a=-2*init_scale, b=2*init_scale)
        self.b2 = nn.Parameter(torch.zeros(1, in_size))

    def forward(self, x):
        # Input shape: [batch_size, seq_len, in_size]
        # Down-project to bottleneck
        net = torch.matmul(x, self.w1) + self.b1
        # GELU activation
        net = F.gelu(net)
        # Up-project back to input dimension
        net = torch.matmul(net, self.w2) + self.b2
        # Residual connection
        return net + x


# from seq2seq.hypercomplex.layers import PHMLinear
# from .low_rank_layer import LowRankLinear
# class LowRankAdapter(nn.Module):
#     """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.input_dim = config.input_dim
#         self.down_sample_size = self.input_dim // config.reduction_factor
#         self.activation = Activations(config.non_linearity.lower())
#         self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
#                                           w_init=config.low_rank_w_init,
#                                           rank=config.low_rank_rank)
#         self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
#                                         w_init=config.low_rank_w_init,
#                                         rank=config.low_rank_rank)

#     def forward(self, x):
#         z = self.down_sampler(x)
#         z = self.activation(z)
#         output = self.up_sampler(z)
#         return output