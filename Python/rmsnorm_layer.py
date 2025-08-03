# Layer onto the cuda and binding files to connect to the actual
# python pytorch function
import torch
import torch.nn as nn
import torch.utils.cpp_extension import load
import os


# JIT compile the exention
_rmsnorm_cuda = load()

# class and functions to replace rmsnorm functions in python
# with my own cuda + cpp implementation
class CustomRMSNorm(nn.module):
    def __init__():
    

    def forward():

    def from_standard_rmsnorm():