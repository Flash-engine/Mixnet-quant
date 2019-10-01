import torch
import networks.mixnet  as mixnet
from thop import profile
from thop import clever_format

model = mixnet.mixnet_m()
input = torch.randn(1, 3, 32, 32)
flops, params = profile(model, inputs=(input, ))
flops1, params1 = clever_format([flops, params], "%.3f")
print("mixnet has {} Flops and {} Parameters".format(flops1, params1))

base_flops = 10490.0e6
base_params = 36.5e6

flops_score = float(flops * 2) / base_flops
params_score = float(params) / base_params
print("mixnet score: {} + {} = {}".format(flops_score, params_score, flops_score + params_score)) 

