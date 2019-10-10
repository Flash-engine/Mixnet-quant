# coding=utf-8
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from quantize.quantize_methods import dsq_quant as quant
import torch.nn.functional as F

'''
@Brief:params in the dict provide intinial values
       per_channel is for weight only
       activation values are quantized per-layer by default
'''
DSQ_QPARAMS={'w_alpha':0.5,\
        'act_alpha':0.5,\
        'act_quant':True,\
        'per_channel':True,\
        'w_qbit':8,\
        'act_qbit':8}

def get_max_min(input,per_channel):
        if not per_channel:  
             result_data_min = torch.min(input)
             result_data_max = torch.max(input)
        else:
             input_data = input.view(input.size(0) , -1)
             data_min = torch.min(input_data,-1)[0]
             data_max = torch.max(input_data,-1)[0]
             shape = [*data_max.shape,*[1] * \
                     ( len(input.shape) - len(data_max.shape) )]
             result_data_min = data_min.view(shape)
             result_data_max = data_max.view(shape)

        return result_data_max, result_data_min



'''
@Brief: Basic nn modules for constructing networks
'''
class QConv2d(nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,qparams=DSQ_QPARAMS):
 
        super(QConv2d, self).__init__(n_channels, out_channels, kernel_size, \
                stride,padding, dilation, groups, bias)

        assert qparams is not None

        ## DSQ params require gradients
        self.w_alpha = Parameter(torch.tensor(qparams['w_alpha'],requires_grad=True))
        self.w_qbit = qparams['w_qbit']

        self.act_alpha = Parameter(torch.tensor(qparams['act_alpha'],requires_grad=True))
        self.act_qbit = qparams['act_qbit']

        self.act_quant = qparams['act_quant']
        self.per_channel = qparams['per_channel']


   
    def forward(self, input):

        w_data_max ,w_data_min = get_max_min(self.weight,self.per_channel)

        qweight = quant(self.weight,self.w_qbit,\
                w_data_max,\
                w_data_min,\
                self.w_alpha)

        if self.act_quant:
            act_data_max ,act_data_min = get_max_min(input,False)
            qinput = quant(input,self.act_qbit,\
                        act_data_max,\
                        act_data_min,\
                        self.act_alpha)
        else:
            qinput=input

        qbias = self.bias #NOTE:we do not consider quantizing bias

        return F.conv2d(qinput, qweight, qbias, self.stride,
                        self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,qparams=DSQ_QPARAMS):

        super(QWLinear, self).__init__(in_features, out_features, bias)

        assert qparams is not None

        ## DSQ params require gradients
        self.w_alpha = Parameter(torch.tensor(qparams['w_alpha'],requires_grad=True))
        self.w_qbit = qparams['w_qbit']

        self.act_alpha = Parameter(torch.tensor(qparams['act_alpha'],requires_grad=True))
        self.act_qbit = qparams['act_qbit']

        self.act_quant = qparams['act_quant']
        self.per_channel = qparams['per_channel']


   
    def forward(self, input):

        w_data_max ,w_data_min = get_max_min(self.weight,self.per_channel)

        qweight = quant(self.weight,self.w_qbit,\
                w_data_max,\
                w_data_min,\
                w_alpha)

        if self.act_quant:
            act_data_max ,act_data_min = get_max_min(input,False)
            qinput = quant(input,self.act_qbit,\
                        act_data_max,\
                        act_data_min,\
                        self.act_alpha)
        else:
            qinput=input

        qbias = self.bias #NOTE:we do not consider quantizing bias

        return F.linear( qinput, qweight, qbias )
