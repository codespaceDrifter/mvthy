import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


import torch
from model.transformer.classic_transformer import ClassicTransformer
import torch.nn as nn
import torch
from utils.bin_to_tensors import bin_to_tensors
from tokenization.tokenizer import Tokenizer
from train.trainer import train_model
from utils.tensor_dataset import TensorDataset
from train.saves import load_latest_checkpoint


def inference(model, tokenizer, max_length=200):
    model.eval()
    model.cuda()  # Move model to CUDA
    
    print("Interactive inference mode. Type 'exit' to quit.")
    
    while True:
        user_input = input("Enter text: ")
        
        if user_input.lower() == 'exit':
            break
        
        x = tokenizer.encode(user_input, add_SOS=False, add_EOS=False).unsqueeze(0).cuda()
        y = tokenizer.encode("", add_SOS=True, add_EOS=False).unsqueeze(0).cuda()
        with torch.no_grad():
            while y.size(-1) < max_length:
                # (batch, 1)
                next_token_ids = model.predict(x, y)

                if torch.all(next_token_ids == tokenizer.EOS_ID):
                    break
                next_token_word = tokenizer.decode(next_token_ids[0].cpu())
                print(next_token_word[0], end="", flush=True)

                y = torch.cat((y, next_token_ids), dim=1)

        print()  # Add newline after generation is complete

tokenizer = Tokenizer(os.path.join(project_root, "tokenization/assets/token_to_id.json"))
vocab_size = len (tokenizer.id_to_token)

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



saves_folder = os.path.join(project_root, "checkpoints", "perma")
load_latest_checkpoint(saves_folder, model)
inference (model, tokenizer, 24)
