import os
import sys


def create_int_addition_txt():
    projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    src_file = os.path.join(projectRoot, "curriculum/0/txt/src.txt")
    tgt_file = os.path.join(projectRoot, "curriculum/0/txt/tgt.txt")
    with open(src_file, "w") as s, open(tgt_file, "w") as g:
        for a in range(-1000, 1001):
            for b in range(-1000, 1001):
                expr = f"{a}+{b}=" if b >= 0 else f"{a}{b}="
                result = f"{a + b}"
                s.write(expr + "\n")
                g.write(result + "\n")
    
    print(f"Done. Output written")
