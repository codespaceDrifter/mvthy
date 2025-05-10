import os
import sys
import json
import string

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_file = os.path.join(projectRoot, "curriculum/txt/int_addition.txt")


sys.path.append(projectRoot)

with open(output_file, "w") as f:
    for a in range(-1000, 1001):
        for b in range(-1000, 1001):
            if b >= 0:
                expr = f"{a}+{b}="
            else:
                expr = f"{a}{b}="
            result = f"{a + b}"
            f.write(expr + "\n" + result + "\n")

print(f"Done. Output written to {output_file}")
