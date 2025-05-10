import torch
import os

import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



# ADD ALL TXT CREATION SCRIPTS HERE TOO. THIS SHOULD CREATE THE ENTIRE DATASET. FROM SCRATCH. GITHUB DOESNT TRACK DATSETS ANYMORE. 

from tokenization.tokenizer import Tokenizer
from utils.txt2bin import tokenize_to_dataset
dataset_root = os.path.join (project_root, "curriculum")

tokenizer = Tokenizer(os.path.join(project_root, "tokenization/assets/token_to_id.json"))
input_path = os.path.join(dataset_root, "txt", "int_addition.txt")
train_path = os.path.join(dataset_root, "bin/int_addition", "train.bin")
test_path= os.path.join(dataset_root, "bin/int_addition", "test.bin")
vlad_path = os.path.join(dataset_root, "bin/int_addition", "vlad.bin")

tokenize_to_dataset(tokenizer = tokenizer,
                        input_path= input_path,
                        train_path= train_path,
                        test_path=test_path,
                        valid_path= vlad_path,
                        input_len = 32,
                        output_len = 16,
                        train_ratio = 0.9,
                        test_ratio = 0.1,
                        valid_ratio = 0)
