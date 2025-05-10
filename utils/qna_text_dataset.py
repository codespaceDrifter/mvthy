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

class TransformerDataset(data.Dataset):
    def __init__(self, path: str,
                 input_len: int,
                 output_len: int,
                 dtype: torch.dtype = torch.int32):
        self.path = path
        self.input_len = input_len
        self.output_len = output_len
        self.dtype = dtype 

        self.stride = input_len + output_len
        self.item_size = (input_len + output_len) * torch.tensor([], dtype=dtype).element_size()
        self.total_tokens = os.path.getsize(path) // torch.tensor([], dtype=dtype).element_size()
        self.length = (self.total_tokens - (input_len + output_len)) // self.stride

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        byte_offset = idx * self.stride * torch.tensor([], dtype=self.dtype).element_size()
        with open(self.path, "rb") as f:
            f.seek(byte_offset)
            buf = f.read(self.item_size)
            tokens = torch.frombuffer(buf, dtype=self.dtype)

        x = tokens[:self.input_len]
        y = tokens[self.input_len:self.input_len + self.output_len]

        return x, y

# testing script


