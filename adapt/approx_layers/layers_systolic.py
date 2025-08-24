# adapt/layers_systolic.py
#for quantization
import torch.utils.data
import pytorch_quantization.utils
import pytorch_quantization.nn.modules._utils as _utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn
#
from torch.utils.cpp_extension import load
from .torch_utils import _ConvNd, _size_2_t, Union, Tensor, Optional, _pair
import torch.nn.functional as F 
import torch.nn as nn
from torch.nn import Parameter
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import math


# Reuse your base modules if you like; here we write compact versions that match your API.

class AdaPT_Linear_Systolic(nn.Module):
    def __init__(self, size_in, size_out, bias=True, axx_mult='mul8s_acc', use_exact=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size_out, size_in))
        self.bias_ = bias
        self.bias = nn.Parameter(torch.empty(size_out)) if bias else None

        # Build + load the systolic kernel
        cflags = [f'-DAXX_MULT={axx_mult}', '-march=native', '-fopenmp', '-O3']
        if use_exact:
            cflags.append('-DUSE_EXACT')
        self.axx_linear_kernel = load(
            name=('PyInit_linear_systolic_exact_' if use_exact else 'PyInit_linear_systolic_') + axx_mult,
            sources=['/workspace/adapt/adapt/cpu-kernels/axx_linear_systolic.cpp'],
            extra_cflags=cflags,
            extra_ldflags=['-lgomp'],
            verbose=False
        )

        # Init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Quant
        num_bits = 8
        unsigned = False
        self.max_value = (2**num_bits - 1) if unsigned else (2**(num_bits-1) - 1)
        qdesc = QuantDescriptor(num_bits=num_bits, fake_quant=False, unsigned=unsigned, calib_method='histogram')
        self.quantizer   = TensorQuantizer(qdesc)
        self.quantizer_w = TensorQuantizer(qdesc)

    def forward(self, x):
        # Accurate path if not yet calibrated
        if self.quantizer.amax is None or self.quantizer_w.amax is None:
            out = x.matmul(self.weight.t())
            return out + self.bias if self.bias_ else out

        qx = self.quantizer(x).to(torch.int8)
        qw = self.quantizer_w(self.weight).to(torch.int8)
        out_i32 = self.axx_linear_kernel.forward(qx, qw)  # [B, out]
        scale = (self.max_value / self.quantizer.amax) * (self.max_value / self.quantizer_w.amax)
        out = out_i32.to(torch.float32) / scale
        return out + self.bias if self.bias_ else out


class AdaPT_Conv2d_Systolic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 axx_mult='mul8s_acc', use_exact=False):
        super().__init__()
        if groups != 1 and groups != in_channels:
            raise ValueError('AdaPT_Conv2d_Systolic supports groups == 1 or depthwise (groups==in_channels)')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = groups
        self.bias_ = bias
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        cflags = [f'-DAXX_MULT={axx_mult}', '-march=native', '-fopenmp', '-O3']
        if use_exact:
            cflags.append('-DUSE_EXACT')
        self.axx_conv2d_kernel = load(
            name=('PyInit_conv2d_systolic_exact_' if use_exact else 'PyInit_conv2d_systolic_') + axx_mult,
            sources=['/workspace/adapt/adapt/cpu-kernels/axx_conv2d_systolic.cpp'],
            extra_cflags=cflags,
            extra_ldflags=['-lgomp'],
            verbose=False
        )

        # init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Quant
        num_bits = 8
        unsigned = False
        self.max_value = (2**num_bits - 1) if unsigned else (2**(num_bits-1) - 1)
        qdesc = QuantDescriptor(num_bits=num_bits, fake_quant=False, unsigned=unsigned, calib_method='histogram')
        self.quantizer   = TensorQuantizer(qdesc)
        self.quantizer_w = TensorQuantizer(qdesc)

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode=self.padding_mode)
            pad = (0,0)
        else:
            pad = self.padding

        # Accurate path until calibration exists
        if self.quantizer.amax is None or self.quantizer_w.amax is None:
            out = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=pad,
                           dilation=self.dilation, groups=self.groups)
            return out

        qx = self.quantizer(x).to(torch.int8)
        qw = self.quantizer_w(self.weight).to(torch.int8)

        # depthwise groups: split & concat using the same kernel (simple + robust)
        if self.groups > 1 and self.groups == self.in_channels and self.out_channels == self.in_channels:
            outs = []
            for c in range(self.in_channels):
                x_c = qx[:, c:c+1]
                w_c = qw[c:c+1]
                o_c = self.axx_conv2d_kernel.forward(
                    x_c, w_c,
                    list(self.kernel_size),
                    list(self.stride),
                    list(pad)
                )
                outs.append(o_c)
            out_i32 = torch.cat(outs, dim=1)
        else:
            out_i32 = self.axx_conv2d_kernel.forward(
                qx, qw,
                list(self.kernel_size),
                list(self.stride),
                list(pad)
            )

        scale = (self.max_value / self.quantizer.amax) * (self.max_value / self.quantizer_w.amax)
        out = out_i32.to(torch.float32) / scale
        return out + self.bias.view(1, -1, 1, 1) if self.bias_ else out
