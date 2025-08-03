import sys
import time
import torch
from pathlib import Path

# add lit llama to path
# include it in this directory
# import llama model
# and import llama utils

# import my custom RMSNorm
from Python.rmsnorm_layer import CustomRMSNorm
# reaplce litllamas rmsnorm with mine
def replace_rmsnorm_in_model():

# benchmark lit llamas inference with
# and without my rmsnorm implementation
    
def benchmark_lit_llama_inference():
    # load model config

    # create model without loading weights (find out why models need weights)

    # replace rmsnorm layers

    # test input

    # warmup

    # benchmark standard model

    # benchmark custom model

    # post/validate results

    # check output similarity, should be identical



# check if current script is being ran as the 
# main program, that means this is the file to run
if __name__ == "__main__":
    result = benchmark_lit_llama_inference()