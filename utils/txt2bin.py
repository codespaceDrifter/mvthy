import torch
import os

import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tokenization.tokenizer import Tokenizer


def tokenize_to_dataset(tokenizer: Tokenizer,
                        input_path: str,
                        train_path: str,
                        test_path: str,
                        valid_path: str,
                        input_len: int,
                        output_len: int,
                        train_ratio: float = 0.9,
                        test_ratio: float = 0.1,
                        valid_ratio: float = 0.0,
                        dtype: torch.dtype = torch.int32):

    assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First pass: just count lines
    print("Counting lines...")
    total_lines = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                total_lines += 1
    
    train_end = int(total_lines * train_ratio)
    test_end = train_end + int(total_lines * test_ratio)
    
    # Second pass: process one line at a time
    processed_lines = 0
    with open(input_path, "r", encoding="utf-8", errors="replace", newline='\n') as in_file, \
         open(train_path, "wb") as train_file, \
         open(test_path, "wb") as test_file, \
         open(valid_path, "wb") as valid_file:


        tokenizing_input = True
        

        for line in in_file:
            assert (line)
            line = line.strip()
            # Process single line
            if tokenizing_input:
                ids = tokenizer.encode(
                    line, add_SOS=False, add_EOS=False, PAD=input_len, PAD_front = True, PAD_back = False).tolist()
            else:
                ids = tokenizer.encode(
                    line, add_SOS=True, add_EOS=True, PAD=output_len, PAD_front = False, PAD_back = True).tolist()

            arr = torch.tensor(ids, dtype=dtype).numpy().tobytes()
            
            # Write to appropriate split
            if processed_lines < train_end:
                train_file.write(arr)
            elif processed_lines < test_end:
                test_file.write(arr)
            else:
                valid_file.write(arr)

            # flip the train / test
            tokenizing_input = not tokenizing_input
                
            processed_lines += 1
            if processed_lines % 1000 == 0:
                print(f"Processed {processed_lines}/{total_lines} lines ({processed_lines/total_lines*100:.2f}%)")
                print(f"Line: {line}")
                print(f"IDs: {ids}")
        
        print(f"✅ Done. Split into {train_end} train, {test_end-train_end} test, {total_lines-test_end} valid examples")



