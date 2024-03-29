import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import pdb
import math
import numpy as np
from mvm import *

import time
# os.environ['CUDA_VISIBLE_DEVICES']='3'
torch.set_printoptions(threshold=10000)

# Custom conv2d formvm function: Doesn't work for back-propagation
class Conv2d_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # +--------------------------+
    # |            MVM           |   
    # +--------------------------+
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind=False):
       
        ## fixed-16: 
        ## sign     : 1 
        ## integer  : 3
        ## fraction : 12
        if weight_bit_frac == -1:
            weight_bit_frac = weight_bits//4*3
        if input_bit_frac == -1:
            input_bit_frac = input_bits//4*3
        if acm_bit_frac == -1:
            acm_bit_frac = acm_bits//4*3
        if adc_bit == -1:
            adc_bit = int(math.log2(XBAR_ROW_SIZE))
            if bit_stream != 1:
                adc_bit += bit_stream
            if bit_slice != 1:
                adc_bit += bit_slice

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
        weight_row = weight.shape[2]
        weight_col = weight.shape[3]

        length = weight_channels_in * weight_row * weight_col
        flatten_weight = torch.zeros(weight_channels_out+1, length).to(device)     #####
        flatten_weight[:-1,:] = weight.reshape((weight_channels_out, length))  ## flatten weights
        flatten_bit_slice_weight = bit_slicing(flatten_weight, weight_bit_frac, bit_slice, weight_bits, device) ## v2: flatten weights --> fixed point --> bit slice -- v1
#        print(flatten_bit_slice_weight)
        # bitsliced weight into 128x128 xbars 
        # xbar_row separates inputs --> results in a same column with different rows will be added later
        xbar_row = math.ceil(flatten_bit_slice_weight.shape[0]/XBAR_ROW_SIZE)
        xbar_col = math.ceil(flatten_bit_slice_weight.shape[1]/XBAR_COL_SIZE)

        weight_xbar = torch.zeros((xbar_row*XBAR_ROW_SIZE, xbar_col*XBAR_COL_SIZE)).to(device)
        weight_xbar[:flatten_bit_slice_weight.shape[0], :flatten_bit_slice_weight.shape[1]] = flatten_bit_slice_weight
        xbars = torch.zeros((xbar_row, xbar_col, XBAR_ROW_SIZE, XBAR_COL_SIZE)).to(device)

        bit_slice_num = weight_bits//bit_slice
        bit_stream_num = input_bits//bit_stream

        bias_addr = [weight_channels_out//int(XBAR_COL_SIZE/bit_slice_num), weight_channels_out%int(XBAR_COL_SIZE/bit_slice_num)]      #####
        for i in range(xbar_row):
            for j in range(xbar_col):
                xbars[i,j] = weight_xbar[i*XBAR_ROW_SIZE:(i+1)*XBAR_ROW_SIZE, j*XBAR_COL_SIZE:(j+1)*XBAR_COL_SIZE]

        input_batch = input.shape[0]
        input_channels = input.shape[1]     # weight_channels_in == input_channels
        input_row = input.shape[2] + padding[0]*2
        input_col = input.shape[3] + padding[1]*2
        input_pad = torch.zeros((input_batch, input_channels, input_row, input_col)).to(device)
        input_pad[:,:,padding[0]:input_row-padding[0],padding[1]:input_col-padding[1]] = input
        pos = torch.ones(input_batch, input_channels, weight_row, weight_col).reshape(input_batch,-1).to(device)
        neg = pos.clone().fill_(0)
        
        output_row = (input_row - weight_row)//stride[0] + 1
        output_col = (input_col - weight_col)//stride[1] + 1 
        output = torch.zeros((input_batch, weight_channels_out, output_row, output_col)).to(device)
        flatten_binary_input = torch.zeros(input_batch, xbars.shape[0]*XBAR_ROW_SIZE, bit_stream_num).to(device)
        flatten_input_sign_temp = torch.zeros(input_batch, xbars.shape[0]*XBAR_ROW_SIZE, bit_stream_num).to(device)
        flatten_input_sign_xbar= torch.zeros(input_batch, xbars.shape[0],XBAR_ROW_SIZE, bit_stream_num).to(device)

        for i in range(output_row):
            for j in range(output_col):
                input_temp = input_pad[:,:, stride[0]*i:stride[0]*i+weight_row, stride[1]*j:stride[1]*j+weight_col].reshape(input_batch,-1)    ## one set of inputs --> flatten: n x 1
                if bit_stream > 1:
                    flatten_input_sign = torch.where(input_temp > 0, pos, neg).expand(bit_stream_num,-1,-1).permute(1,2,0)
                    flatten_input_sign_temp[:,:flatten_input_sign.shape[1]] = flatten_input_sign
                    flatten_input_sign_xbar = flatten_input_sign_temp.reshape(input_batch, xbars.shape[0],XBAR_ROW_SIZE, bit_stream_num)
                    input_temp.abs_()

                flatten_binary_input_temp = float_to_16bits_tensor(input_temp, input_bit_frac, bit_stream, input_bits, device)   # batch x n x 16
#                print(flatten_binary_input_temp)
                flatten_binary_input[:,:flatten_binary_input_temp.shape[1]] = flatten_binary_input_temp
                flatten_binary_input_xbar = flatten_binary_input.reshape((input_batch, xbars.shape[0],XBAR_ROW_SIZE, bit_stream_num))
                if ind == True:
                    xbars_out  = mvm_tensor_ind(model, flatten_binary_input_xbar, flatten_input_sign_xbar, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device)   
                else:
                    xbars_out = mvm_tensor(flatten_binary_input_xbar, flatten_input_sign_xbar, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device)
                output[:,:,i,j] += xbars_out[:, :weight_channels_out]

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) 
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
            
        return grad_input, grad_weight, grad_bias, None, None, None, None


class _ConvNd_mvm(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'bit_slice', 'bit_stream','weight_bits', 'weight_bit_frac','input_bits', 'input_bit_frac','adc_bit','acm_bits', 'acm_bit_frac', 'ind']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, ind, check_grad=False):
        super(_ConvNd_mvm, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.bit_slice = bit_slice
        self.bit_stream = bit_stream
        self.weight_bits = weight_bits
        self.weight_bit_frac = weight_bit_frac
        self.input_bits = input_bits
        self.input_bit_frac = input_bit_frac
        self.adc_bit = adc_bit
        self.acm_bits = acm_bits
        self.acm_bit_frac = acm_bit_frac
        self.ind = ind

        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            self.weight = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2d_mvm(_ConvNd_mvm):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', check_grad=False, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d_mvm, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, ind)
    #@weak_script_method
    def forward(self, input):
            return Conv2d_mvm_function.apply(input, self.weight, self.bias, self.stride,
                   self.padding, self.dilation, self.groups, self.bit_slice, self.bit_stream, self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.adc_bit, self.acm_bits, self.acm_bit_frac, self.ind)


class Linear_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind=False):

        if weight_bit_frac == -1:
            weight_bit_frac = weight_bits//4*3
        if input_bit_frac == -1:
            input_bit_frac = input_bits//4*3
        if acm_bit_frac == -1:
            acm_bit_frac = acm_bits//4*3      
        if adc_bit == -1:

            adc_bit = int(math.log2(XBAR_ROW_SIZE))
            if bit_stream != 1:
                adc_bit += bit_stream
            if bit_slice != 1:
                adc_bit += bit_slice

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
        weight_bias = torch.zeros(weight_channels_out+1, weight_channels_in).to(device)
        weight_bias[:-1,:] = weight

        bit_slice_weight = bit_slicing(weight_bias, weight_bit_frac, bit_slice, weight_bits, device) ## v2: flatten weights --> fixed point --> bit slice -- v1
        # bitsliced weight into 128x128 xbars 
        # xbar_row separates inputs --> results in a same column with different rows will be added later
        xbar_row = math.ceil(bit_slice_weight.shape[0]/XBAR_ROW_SIZE)
        xbar_col = math.ceil(bit_slice_weight.shape[1]/XBAR_COL_SIZE)

        weight_xbar = torch.zeros((xbar_row*XBAR_ROW_SIZE, xbar_col*XBAR_COL_SIZE)).to(device)
        weight_xbar[:bit_slice_weight.shape[0], :bit_slice_weight.shape[1]] = bit_slice_weight
        xbars = torch.zeros((xbar_row, xbar_col, XBAR_ROW_SIZE, XBAR_COL_SIZE)).to(device)

        bit_slice_num = weight_bits/bit_slice
        bit_stream_num = input_bits//bit_stream

        bias_addr = [weight_channels_out//int(XBAR_COL_SIZE/bit_slice_num), weight_channels_out%int(XBAR_COL_SIZE/bit_slice_num)]      #####
        for i in range(xbar_row):
            for j in range(xbar_col):
                xbars[i,j] = weight_xbar[i*XBAR_ROW_SIZE:(i+1)*XBAR_ROW_SIZE, j*XBAR_COL_SIZE:(j+1)*XBAR_COL_SIZE]

        input_batch = input.shape[0]
        input_channels = input.shape[1]     # weight_channels_in == input_channels
        pos = torch.ones(input.shape).to(device)
        neg = pos.clone().fill_(0)      

        binary_input = torch.zeros(input_batch, xbars.shape[0]*XBAR_ROW_SIZE, bit_stream_num).to(device)
        input_sign_temp = torch.zeros(input_batch, xbars.shape[0]*XBAR_ROW_SIZE, bit_stream_num).to(device)
        input_sign_xbar = torch.zeros(input_batch, xbars.shape[0],XBAR_ROW_SIZE, bit_stream_num).to(device)

        if bit_stream > 1:
            input_sign = torch.where(input > 0, pos, neg).expand(bit_stream_num, -1, -1).permute(1,2,0)
            input_sign_temp[:,:input_sign.shape[1]] = input_sign
            input_sign_xbar = input_sign_temp.reshape(input_batch, xbars.shape[0],XBAR_ROW_SIZE, bit_stream_num)
            input.abs_()

        binary_input[:,:input.shape[1]] = float_to_16bits_tensor(input, input_bit_frac, bit_stream, input_bits, device)   # batch x n x 16

        binary_input = binary_input.reshape((input_batch, xbars.shape[0], XBAR_ROW_SIZE, bit_stream_num))
        if ind == True:
            xbars_out = mvm_tensor_ind(model, binary_input, input_sign_xbar, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device)
        else:
            xbars_out = mvm_tensor(binary_input, input_sign_xbar, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device)

        output = xbars_out[:, :weight_channels_out]
 
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class Linear_mvm(nn.Module):
    def __init__(self, input_features, output_features, bias=True, bit_slice = 2, bit_stream = 1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind = False):
        super(Linear_mvm, self).__init__()
        self.in_features = input_features
        self.input_features = input_features
        self.out_features = output_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
            self.bias.data.uniform_(-0.1, 0.1) 
        else:
            self.register_parameter('bias', None)

        self.bit_slice = bit_slice
        self.bit_stream = bit_stream
        self.weight_bits =weight_bits
        self.weight_bit_frac = weight_bit_frac
        self.input_bits = input_bits
        self.input_bit_frac = input_bit_frac
        self.adc_bit = adc_bit
        self.acm_bits = acm_bits
        self.acm_bit_frac = acm_bit_frac
        self.ind = ind

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return Linear_mvm_function.apply(input, self.weight, self.bias, self.bit_slice, self.bit_stream, self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.adc_bit, self.acm_bits, self.acm_bit_frac, self.ind)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
