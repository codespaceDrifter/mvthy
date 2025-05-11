import torch
import os

import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



# ADD ALL TXT CREATION SCRIPTS HERE TOO. THIS SHOULD CREATE THE ENTIRE DATASET. FROM SCRATCH. GITHUB DOESNT TRACK DATSETS ANYMORE. 
from utils.temp_scripts.int_add import create_int_addition_txt

create_int_addition_txt()


from tokenization.tokenizer import Tokenizer
from utils.txt_to_bin import txt_to_bin

dataset_root = os.path.join (project_root, "curriculum", "int_addition")

tokenizer = Tokenizer(os.path.join(project_root, "tokenization/assets/token_to_id.json"))

txt_to_bin(tokenizer = tokenizer,
            src_txt_path= os.path.join(dataset_root, "txt", "src.txt"),
            tgt_txt_path= os.path.join(dataset_root, "txt", "tgt.txt"),
            src_bin_path= os.path.join(dataset_root, "bin", "src.bin"),
            tgt_bin_path= os.path.join(dataset_root, "bin", "tgt.bin"),
            input_len= 32,
            output_len= 16)



