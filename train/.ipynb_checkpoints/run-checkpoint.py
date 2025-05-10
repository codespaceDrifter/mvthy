import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.transformer.classic_transformer import ClassicTransformer
import torch.nn as nn
import torch


from utils.qna_text_dataset import TransformerDataset

dataset_root = os.path.join (project_root, "curriculum/bin/int_addition")

train_dataset = TransformerDataset(
    path=os.path.join(dataset_root, "train.bin"),
    input_len=32,
    output_len=16,
)

test_dataset = TransformerDataset(
    path=os.path.join(dataset_root, "test.bin"),
    input_len=32,
    output_len=16,
)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

from tokenization.tokenizer import Tokenizer


tokenizer = Tokenizer(os.path.join(project_root, "tokenization/assets/token_to_id.json"))

vocab_size = len (tokenizer.id_to_token)
print ("vocab size ", vocab_size)

print(f"Train dataset sample decoded input: ", tokenizer.decode(train_dataset[0][0]))
print(f"Train dataset sample decoded target: ", tokenizer.decode(train_dataset[0][1]))

model = ClassicTransformer(
    vocab_size = vocab_size,
    d_model = 512, 
    num_heads = 16,
    num_encoders = 4,
    num_decoders = 4,
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



from train.trainer import train_model

train_model(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    batch_size=8,
    save_folder_path=os.path.join(project_root, "checkpoints"),
    perma_save_folder_path=os.path.join(project_root, "checkpoints/perma"),
    loss_fn=nn.CrossEntropyLoss(ignore_index=0),
    tokenizer=tokenizer,
    batch_per_save=1000,
    clip_grad_norm = 1
)
