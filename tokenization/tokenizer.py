import torch
from tokenization.trie import Trie
import json

class Tokenizer:
    def __init__(self, token_to_id_path):
        with open(token_to_id_path, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.trie = Trie(set(self.token_to_id.keys()))
        self.SOS_ID = self.token_to_id.get('SOS', None)
        self.EOS_ID = self.token_to_id.get('EOS', None)
        self.UNK_ID = self.token_to_id.get('UNK', None)
        self.PAD_ID = self.token_to_id.get('PAD', None)

    def encode(self, text, add_SOS=False, add_EOS=False,pad=True, pad_len = 0):

        # Tokenize and convert to IDs
        tokens = self.trie.tokenize(text)
        ids = [self.token_to_id.get(tok, self.UNK_ID) for tok in tokens]
        # Get the original length
        original_len = len(ids)
        # Add SOS token if required
        if add_SOS:
            ids = [self.SOS_ID] + ids
        # Add EOS token if required
        if add_EOS:
            ids = ids + [self.EOS_ID]
        # Handle padding
        if pad:
            # Ensure PAD is not less than the current length
            assert pad_len >= len(ids), f"PAD size ({pad_len}) is less than sequence length ({len(ids)})"
            # Calculate padding amount
            pad_amount = pad_len - len(ids)
            # Pad AFTER
            ids = ids + [self.PAD_ID] * pad_amount
        return torch.tensor(ids, dtype=torch.int32)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        tokens = [self.id_to_token.get(i, 'UNK') for i in ids]
        return ''.join(tokens).replace('PAD', '').replace('SOS', '').replace('EOS', '')
