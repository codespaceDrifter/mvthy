import torch
import os

import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tokenization.tokenizer import Tokenizer


def txt_to_bin(tokenizer: Tokenizer,
                train_ratio: float,
                src_txt_path: str,
                tgt_txt_path: str,
                src_train_bin_path: str,
                tgt_train_bin_path: str,
                src_test_bin_path: str,
                tgt_test_bin_path: str,
                input_len: int,
                output_len: int,
                dtype: torch.dtype = torch.int32):
    
    # First pass: just count lines
    print("Counting lines...")
    total_lines = 0
    with open(src_txt_path, "r", encoding="utf-8") as src_txt:
        for line in src_txt:
            if line.strip():
                total_lines += 1

    train_lines = int(total_lines * train_ratio)
                
    # Second pass: process one line at a time
    processed_lines = 0
    with open(src_txt_path, "r", encoding="utf-8", newline='\n') as src_txt, \
         open(tgt_txt_path, "r", encoding="utf-8", newline='\n') as tgt_txt, \
         open(src_train_bin_path, "wb") as src_train_bin, \
         open(tgt_train_bin_path, "wb") as tgt_train_bin, \
         open(src_test_bin_path, "wb") as src_test_bin, \
         open(tgt_test_bin_path, "wb") as tgt_test_bin:


        tokenizing_input = True
        

        for src_line, tgt_line in zip(src_txt, tgt_txt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            assert (src_line and tgt_line)

            src_ids = tokenizer.encode(
                src_line, add_SOS=False, add_EOS=False, pad=True, pad_len =input_len).tolist()

            tgt_ids = tokenizer.encode(
                tgt_line, add_SOS=True, add_EOS=True, pad=True, pad_len=output_len).tolist()

            if processed_lines < train_lines:
                src_train_bin.write(torch.tensor(src_ids, dtype=dtype).numpy().tobytes())
                tgt_train_bin.write(torch.tensor(tgt_ids, dtype=dtype).numpy().tobytes())
            else:
                src_test_bin.write(torch.tensor(src_ids, dtype=dtype).numpy().tobytes())
                tgt_test_bin.write(torch.tensor(tgt_ids, dtype=dtype).numpy().tobytes())


            processed_lines += 1
            if processed_lines % 1000 == 0:
                print(f"Processed {processed_lines}/{total_lines} lines ({processed_lines/total_lines*100:.2f}%)")



