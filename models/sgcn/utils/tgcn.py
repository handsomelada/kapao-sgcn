# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    r"""
    The basic module for applying a graph convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        # n, c, v = x.size()
        x = torch.einsum('ncv,kvw->ncw', (x, A))
        # print('xxxxxx einsum', x.size())

        return x.contiguous(), A
