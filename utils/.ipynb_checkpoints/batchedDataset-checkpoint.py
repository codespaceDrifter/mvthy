import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from typing import Tuple, Optional

class BatchedDataset(data.Dataset):
    def __init__(self, 
                 src_path: str, 
                 tgt_path: str, 
                 input_len: int, 
                 output_len: int,
                 batch_size: int = 128,
                 dtype: torch.dtype = torch.int32):
        """
        Dataset that loads data in batches from binary files using memory mapping.
        
        Args:
            src_path: Path to source binary file
            tgt_path: Path to target binary file
            input_len: Length of each input sequence
            output_len: Length of each output sequence
            batch_size: Size of batches to return
            dtype: Data type of stored values
        """
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.dtype = dtype
        
        # Calculate number of samples
        src_size = os.path.getsize(src_path)
        tgt_size = os.path.getsize(tgt_path)
        
        self.src_item_bytes = np.dtype(self._numpy_dtype_from_torch(dtype)).itemsize * input_len
        self.tgt_item_bytes = np.dtype(self._numpy_dtype_from_torch(dtype)).itemsize * output_len
        
        self.src_samples = src_size // self.src_item_bytes
        self.tgt_samples = tgt_size // self.tgt_item_bytes
        
        assert self.src_samples == self.tgt_samples, "Source and target files must have compatible number of samples"
        
        self.total_samples = self.src_samples
        self.num_batches = self.total_samples // batch_size
        
        # Create memory maps
        numpy_dtype = self._numpy_dtype_from_torch(dtype)
        self.src_mmap = np.memmap(src_path, dtype=numpy_dtype, mode='r', 
                                 shape=(self.total_samples, input_len))
        self.tgt_mmap = np.memmap(tgt_path, dtype=numpy_dtype, mode='r', 
                                 shape=(self.total_samples, output_len))
    
    def _numpy_dtype_from_torch(self, torch_dtype):
        """Convert torch dtype to numpy dtype"""
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
        }
        return dtype_map.get(torch_dtype, np.float32)
    
    def __len__(self):
        """Returns the number of batches."""
        return self.num_batches
    
    def __getitem__(self, idx):
        """Get a batch by index."""
        if idx >= self.num_batches:
            raise IndexError("Index out of bounds")
        
        # Calculate start and end indices
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        
        # Extract the batch and convert to torch tensors
        src_batch = torch.from_numpy(self.src_mmap[start_idx:end_idx].copy())
        tgt_batch = torch.from_numpy(self.tgt_mmap[start_idx:end_idx].copy())
        
        # Ensure tensors don't require gradients
        src_batch.requires_grad_(False)
        tgt_batch.requires_grad_(False)
        
        return src_batch, tgt_batch

# used in dataloader
def squeeze_batch_collate(batch):
    """
    Custom collate function that squeezes away the batch dimension.
    Assumes batch contains a single item.
    """
    assert len(batch) == 1, "This collate function only works with batch_size=1"
    batch_item = batch[0]
    return batch_item



