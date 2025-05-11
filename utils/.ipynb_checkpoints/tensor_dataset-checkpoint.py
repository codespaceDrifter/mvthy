import torch
from torch.utils.data import Dataset, DataLoader
import math


class TensorDataset(Dataset):
    def __init__(self, src, tgt):
        """
        A dataset that serves pre-loaded tensors.
        
        Args:
            src: Source tensor [num_samples, ...] 
            tgt: Target tensor [num_samples, ...]
        """
        assert src.shape[0] == tgt.shape[0], "Source and target must have same first dimension"
        
        self.src = src
        self.tgt = tgt
        self.num_samples = src.shape[0]
    
    def __len__(self):
        """Returns the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample by index."""
        return self.src[idx], self.tgt[idx]