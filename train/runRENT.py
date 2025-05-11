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
from utils.tensor_dataset import TensorDataset

dataset_root = os.path.join (project_root, "curriculum/int_addition")

src_train, tgt_train, src_test, tgt_test = bin_to_tensors(
    src_path = os.path.join(dataset_root, "bin", "src.bin"),
    tgt_path = os.path.join(dataset_root, "bin", "tgt.bin"), 
    input_len = 32,
    output_len = 16,
    train_test_ratio= 0.8,
    dtype = torch.int32)

print ("src train shape", src_train.shape)
print ("tgt train shape", tgt_train.shape)
print ("src test shape", src_test.shape)
print ("tgt test shape", tgt_test.shape)

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
    d_ff = 2048,
    loss_fn = nn.CrossEntropyLoss(ignore_index=0),
    max_seq_len = 10000,
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


train_dataset = TensorDataset(src_train, tgt_train)
test_dataset = TensorDataset(src_test, tgt_test)

train_model(
    train_dataset = train_dataset,
    test_dataset = test_dataset,
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
