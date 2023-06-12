import tvm
from models.vicuna_v1_7b_fp16 import Module as Vicuna
from extraction import extract_from_relax

if __name__ == "__main__":
    extract_from_relax(mod=Vicuna, model_name="vicuna_v1_7b_fp_16")
