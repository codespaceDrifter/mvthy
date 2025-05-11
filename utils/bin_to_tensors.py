import os
import sys
import json
import string

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import torch.utils.data as data
import numpy as np
from typing import List, Tuple, Dict, Optional

import os
import torch
import random
from typing import Tuple

def bin_to_tensors(src_path: str, tgt_path: str, 
                    input_len: int, output_len: int,
                    train_test_ratio: float = 0.8,
                    dtype: torch.dtype = torch.int32):

    # Load the entire files into memory as tensors
    src_tensor = torch.frombuffer(open(src_path, 'rb').read(), dtype=dtype)
    tgt_tensor = torch.frombuffer(open(tgt_path, 'rb').read(), dtype=dtype)
    
    # Calculate how many complete samples we can make
    src_samples = src_tensor.numel() // input_len
    tgt_samples = tgt_tensor.numel() // output_len
    assert (src_samples == tgt_samples)
    
    # Reshape tensors to get samples
    src_all = src_tensor.view(-1, input_len)
    tgt_all = tgt_tensor.view(-1, output_len)
    
    # Calculate split sizes
    train_size = int(src_samples * train_test_ratio)
    
    # Create a random permutation for shuffling
    indices = torch.randperm(src_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Split the data
    src_train = src_all[train_indices].clone()
    tgt_train = tgt_all[train_indices].clone()
    src_test = src_all[test_indices].clone()
    tgt_test = tgt_all[test_indices].clone()

    #no grad
    src_train.requires_grad_(False) 
    tgt_train.requires_grad_(False) 
    src_test.requires_grad_(False) 
    tgt_test.requires_grad_(False) 
    
    return src_train, tgt_train, src_test, tgt_test

