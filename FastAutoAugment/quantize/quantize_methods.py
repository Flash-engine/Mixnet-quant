# coding=utf-8

import torch
import math
import numpy as np
from torch.autograd.function import  Function

'''
@Brief:math operations for dorefa quantization functions
'''
def quantize_tanh(input,qbit):
    n = math.pow(2.0, qbit) - 1
    return torch.round(i * n) / n

def quantize_gemm(input,qbit):
        n = math.pow(2.0, qbit) - 1
        v_max = torch.max(i)
        v_min = torch.min(i)
        scale = (v_max - v_min)/n
        scale = max(scale, 1e-8)
        zero_point = torch.round(torch.clamp(-v_min/scale, 0, n))
        quantize_val = torch.clamp(torch.round(i/scale + zero_point), 0, n)
        return (quantize_val-zero_point) * scale

'''
@Brief:Quntizaion fucntions for dorefa
'''
class QuantizeFunc(Function):

    '''
    @brief:init the quant function with a qparam dict 
           qparam dict specifies the way to quant
    @param[in]:qparams={'symmetric':,\
                        'per-channel':,\
                        'qbit':,\
                        }
    '''
    @staticmethod
    def forward(ctx,input,qparams):
        pass

    @staticmethod
    def backward(ctx,grad_outputs):
        pass


'''
@Brief:Quant fucntions for DSQ
'''
K_MAX=1e10
class DSQ_QuantFunc(Function):
      @staticmethod
      def forward(ctx,input,qbit,data_max,data_min,alpha):
        bit_width = qbit
        l = data_min
        u = data_max

        l = l.view(*l.shape, *[1] * (len(input.shape) - len(l.shape)))#NOTE
        u = u.view(*u.shape, *[1] * (len(input.shape) - len(u.shape)))#NOTE

        output = input.clone()
        step = (u - l) / (2**bit_width - 1)

        scale = 1.0 / (1.0 - alpha)
        k = torch.log((scale + 1) / (scale - 1)) / step
        k.clamp_(max=K_MAX)

        idx = torch.floor( (output - l) / step )
        m = l + step * (idx + 0.5)
        l_mask = output.lt( l )
        u_mask = output.gt( u )

        z = output - m

        sq = torch.tanh(z*k)#NOTE:soft-quant

        # Q(x) with repect to x if x lie in (l,u)
        x_derivate = step * 0.5 * scale * k * (1 - sq * sq)
        
        # k with respect to alpha,l and u
        k_alpha_derivate = 1 / (step * torch.log( torch.tensor(2.0) ) ) * 2/(alpha**2-2*alpha)
        k_l_derivate = -1 / (step**2) * torch.log(2/alpha-1) / (1-2**bit_width)
        k_u_derivate = -1 / (step**2) * torch.log(2/alpha-1) / (2**bit_width-1)
       
        # fi with repect to alpha
        fi_alpha_derivate = sq + (1-alpha) * (1-sq*sq) * k_alpha_derivate * z
        fi_alpha_derivate /= (1-alpha)**2
        
        # fi with respect to l
        fi_l_derivate = 1/(1-alpha)*(1-sq**2)
        fi_l_derivate *= ( k_l_derivate*z - k*( 1+ (idx+0.5)/(1-2**bit_width) ) ) 

        # fi with respect to u
        fi_u_derivate = 1/(1-alpha)*(1-sq**2)
        fi_u_derivate *= ( k_u_derivate*z - k*(idx+0.5)/(2**bit_width-1) )

        sq = scale * sq

        # Q(x) with respect to alpha,l and u
        q = idx + 0.5*(sq+1)
        l_derivate = 1 + q/(1-2**bit_width) + step * 0.5 * fi_l_derivate 
        u_derivate = q/(2**bit_width-1) + step * 0.5 * fi_u_derivate 
        alpha_derivate = step*0.5*fi_alpha_derivate
        
        sq=torch.sign(sq)
        output = idx + 0.5 * (sq + 1)#NOTE:de-quant
        output = l + step * output
        output = torch.max(output, l)
        output = torch.min(output, u)

        ctx.save_for_backward(input,idx, l_mask, u_mask, \
                x_derivate,alpha_derivate,l_derivate,u_derivate)

        return output

      @staticmethod
      def backward(ctx,grad_output):
          input, idx, l_mask, u_mask, \
                  x_derivate,alpha_derivate,l_derivate,u_derivate = ctx.saved_tensors
          
          # compute l gradient
          l_derivate[l_mask] = 1
          l_derivate[u_mask] = 0
          grad_data_min = grad_output * l_derivate 
          
          # compute u gradient
          u_derivate[l_mask] = 0
          u_derivate[u_mask] = 1
          grad_data_max = grad_output * u_derivate

          # compute alpha gradient
          grad_alpha = grad_output * alpha_derivate
          grad_alpha[l_mask] = 0
          grad_alpha[u_mask] = 0


          grad_input = grad_output * x_derivate
          grad_input[l_mask] = 0
          grad_input[u_mask] = 0

          return grad_input,None,grad_data_max,grad_data_min,grad_alpha


dorefa_quant = QuantizeFunc.apply

dsq_quant = DSQ_QuantFunc.apply
