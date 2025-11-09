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

from torch.utils.cpp_extension import load


class AdaPT_Linear_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bias_, quantizer, quantizer_w, amax, amax_w, max_value, axx_linear_kernel):
        ctx.save_for_backward(input, weight, bias)
        ctx.bias_ = bias_
        
        quant_weight = quantizer_w(weight)
        quant_input = quantizer(input)   
                                
        if (amax == None):
            output = input.mm(weight.t())
        else:     
            quant_input = quant_input.to(dtype=torch.int8)
            quant_weight = quant_weight.to(dtype=torch.int8)
            
            output = axx_linear_kernel.forward(quant_input, quant_weight)
            output = output/((max_value/amax)*(max_value/amax_w))                           
                                      
        if bias_:
            return output+bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        bias_ = ctx.bias_
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias_ and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class AdaPT_Linear(nn.Module):
    def __init__(self, size_in, size_out, bias=True, axx_mult='mul8s_acc'):
        
        super(AdaPT_Linear, self).__init__()
        
        self.size_in, self.size_out, self.bias_ = size_in, size_out, bias
        self.fn = AdaPT_Linear_Function.apply
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        self.axx_mult = axx_mult
        
        #Jit compilation method for cpp extention
        #set PyInit_ prefix to comply with the python module name
        self.axx_linear_kernel = load(name='PyInit_linear_'+axx_mult, sources=["/workspace/adapt/adapt/cpu-kernels/axx_linear.cpp"], extra_cflags = ['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3' ], extra_ldflags=['-lgomp'], verbose=True)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
               
        num_bits=8
        unsigned=False
        if unsigned: 
            self.max_value = pow(2,num_bits)-1
        else:
            self.max_value = pow(2,num_bits-1)-1

        self.quant_desc = QuantDescriptor(num_bits=num_bits, fake_quant=False, unsigned=unsigned, calib_method='histogram')
        self.quantizer = TensorQuantizer(self.quant_desc)
        self.quantizer_w = TensorQuantizer(self.quant_desc)
       
    def forward(self, x):       
        x = self.fn(x, self.weight, self.bias, self.bias_, self.quantizer, self.quantizer_w, self.quantizer.amax, self.quantizer_w.amax, self.max_value, self.axx_linear_kernel)
        
        return x
     
class AdaPT_Conv2d_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, quantizer, quantizer_w, kernel_size, amax, amax_w, max_value, out_channels, bias_, axx_conv2d_kernel, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode = 'zeros'):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
                         
        if padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=padding_mode),
                            weight, bias, stride,
                            _pair(0), dilation, groups)
                    
        quant_weight = quantizer_w(weight)
        quant_input = quantizer(input)    
        
        #in case of normal padding_mode run approx_conv2d
        #TODO quant case for bias=True. Currently not needed for typical ConvNets
        #quant descriptors inside class is slower than using quant_nn.Conv2d instead
                        
        if (amax == None):           
            return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        else:
            
            #convert weights and biases to int8 (must change if approx_mult bits are higher)
            quant_input = quant_input.to(dtype=torch.int8)
            quant_weight = quant_weight.to(dtype=torch.int8)
             
            #slow temporary version of grouped conv2d using simple conv2d (split+concat)
            #support only for group = input_dim = outputdim (i.e. mobilenetv2)
            if groups > 1 : 
                out=torch.empty(0)
                for i in range(0,groups):
                    filters = quant_weight[i:(i+1)]                   
                    o =  axx_conv2d_kernel.forward(quant_input[:, i:(i+1)], filters, kernel_size, stride, padding) 
                    out = torch.cat((out, o), dim=1)
                    
                out = (out/((max_value/amax)*((max_value/amax_w))))
                if bias_:
                    return out + bias.reshape(1,out_channels,1,1)   
                else: 
                    return out
            
            if bias_:               
                out = axx_conv2d_kernel.forward(quant_input, quant_weight, kernel_size, stride, padding)                
                return (out/((max_value/amax)*((max_value/amax_w))))+bias.reshape(1,out_channels,1,1)
            
            out = axx_conv2d_kernel.forward(quant_input, quant_weight, kernel_size, stride, padding)
      
            return out/((max_value/amax)*((max_value/amax_w)))

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None            

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class AdaPT_Conv2d(_ConvNd):
    def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: Union[str, _size_2_t] = 0,
    dilation: _size_2_t = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',  # TODO: refine this type
    axx_mult='mul8s_acc',
    device=None,
    dtype=None   
    ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        self.bias_ = bias #added for case of none bias in order to run conv2d-cpu kernel
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.axx_mult = axx_mult
        
        super(AdaPT_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        
        #Jit compilation method for cpp extention
        #set PyInit_ prefix to comply with the python module name
        self.axx_conv2d_kernel = load(name='PyInit_conv2d_'+axx_mult, sources=["/workspace/adapt/adapt/cpu-kernels/axx_conv2d.cpp"], extra_cflags = ['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3' ], extra_ldflags=['-lgomp'], verbose=True)
        
        if groups != 1 and groups != in_channels:
            raise ValueError('AdaPT_Conv2d does not support groups != in_channels')

        #quantization
        num_bits=8
        unsigned=False
        if unsigned: 
            self.max_value = pow(2,num_bits)-1
        else:
            self.max_value = pow(2,num_bits-1)-1

        self.quant_desc = QuantDescriptor(num_bits=num_bits, fake_quant=False, unsigned=unsigned, calib_method='histogram')
        self.quantizer = TensorQuantizer(self.quant_desc)
        self.quantizer_w = TensorQuantizer(self.quant_desc)


    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):       
    
        return AdaPT_Conv2d_Function.apply(input, weight,  self.quantizer, self.quantizer_w, self.kernel_size, self.quantizer.amax, self.quantizer_w.amax, self.max_value, self.out_channels, self.bias_, self.axx_conv2d_kernel, bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

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
