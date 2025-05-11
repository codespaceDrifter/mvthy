import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.transformer.classic_transformer import ClassicTransformer
import torch.nn as nn
import torch
from utils.bin_to_tensors import bin_to_tensors
from tokenization.tokenizer import Tokenizer
from train.trainer import train_model
from utils.tensor_dataset import BatchedDataset, squeeze_batch_collate
from torch.utils.data import DataLoader


dataset_root = os.path.join (project_root, "curriculum/int_addition")

train_dataset = BatchedDataset (
    src_path = os.path.join(dataset_root, "bin", "src_train.bin"),
    tgt_path = os.path.join(dataset_root, "bin", "tgt_train.bin"),
    input_len = 32,
    output_len = 16,
    batch_size = 256
)

test_dataset = BatchedDataset (
    src_path = os.path.join(dataset_root, "bin", "src_test.bin"),
    tgt_path = os.path.join(dataset_root, "bin", "tgt_test.bin"),
    input_len = 32,
    output_len = 16,
    batch_size = 256
)


train_loader = DataLoader(
    train_dataset, 
    batch_size=1,  # Must be 1 for this approach
    shuffle=True,
    collate_fn=squeeze_batch_collate,
    num_workers=4,
    pin_memory=True,  # for faster GPU transfer
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=1,  # Must be 1 for this approach
    shuffle=True,
    collate_fn=squeeze_batch_collate,
    num_workers=4,
    pin_memory=True,  # for faster GPU transfer
)

    


tokenizer = Tokenizer(os.path.join(project_root, "tokenization/assets/token_to_id.json"))

vocab_size = len (tokenizer.id_to_token)
print ("vocab size ", vocab_size)

print(f"Train dataset sample decoded input: ", tokenizer.decode(src_train[0]))
print(f"Train dataset sample decoded target: ", tokenizer.decode(tgt_train[0]))

model = ClassicTransformer(
    vocab_size = vocab_size,
    d_model = 768, 
    num_heads = 16,
    num_encoders = 6,
    num_decoders = 6,
    d_ff = 3072,
    loss_fn = nn.CrossEntropyLoss(ignore_index=0),
    max_seq_len = 2000,
    pad_id = 0,
    sos_id = 1,
    eos_id = 2,
    unk_id = 3,
    dropout=0.1
)

print (model)


parameter_num = sum(p.numel() for p in model.parameters() )
# 385 M
print(f"parameters num: {parameter_num:,}")


train_model(
    train_loader = train_loader,
    test_loader = test_loader,
    model = model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    batch_size = 3072,
    save_folder_path=os.path.join(project_root, "checkpoints"),
    perma_save_folder_path=os.path.join(project_root, "checkpoints/perma"),
    loss_fn=nn.CrossEntropyLoss(ignore_index=0),
    tokenizer=tokenizer,
    batch_per_save=100,
    clip_grad_norm = 1,
    num_workers = 4,
    debug = False
)
