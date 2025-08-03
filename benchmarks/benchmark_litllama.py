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
# this would make this file the main script to run
def benchmark_lit_llama_inference():
