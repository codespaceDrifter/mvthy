import os
import sys
import json
import string

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(projectRoot)

script_dir = os.path.dirname(__file__)
SAVE_FILE = os.path.join(script_dir, "assets", "token_to_id.json")

# 1. Special tokens
special = ["PAD", "SOS", "EOS", "UNK"]

# 2. Digits with space
digits = [str(i) for i in range(10)]

# 3. Letters
letters = ["a", "b", "c", "x", "y", "z"]

# 4. Punctuation
punc = ["." , "+", "-", "*", "/", "^",  "=", "<", ">", "(", ")",  "{", "}"]

# 5. LaTeX math tokens (space suffixed, no {} ones)
# NOT all latex tokens. very reduced list. might add later
# NOT included: set theory, greek letters, probabilities, etc. 
latex_tokens = [
    # Arithmetic / Algebra
    "\\mod","\\frac",

    # Calculus
    "\\partial", "\\int",

    # Functions
    "\\sin", "\\cos", "\\tan", "\\log", "\\max", "\\min",
]

# Add a space suffix to each LaTeX token
latex_tokens = [tok + " " for tok in latex_tokens]

# Combine all tokens in the required order
all_tokens = special + digits + letters + punc + latex_tokens

# Build token-to-id mapping
token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}

# Save to JSON
os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
with open(SAVE_FILE, "w") as f:
    json.dump(token_to_id, f, indent=2)

print(f"Saved {len(token_to_id)} tokens to {SAVE_FILE}")
