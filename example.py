from models.vicuna_v1_7b_fp16 import Module as Vicuna
from mlc_bench.extraction import extract_from_relax
from mlc_bench.benchmark import MLCBench
import tvm

if __name__ == "__main__":
    # extract_from_relax(
    #     mod=Vicuna, model_name="vicuna_v1_7b_fp_16", file_path="./extracted"
    # )
    MLCBench.benchmark_relax_func(
        mod=Vicuna,
        sample_number=2,
        relax_func="encoding",
        target="llvm -num-cores=10",
        dev=tvm.cpu(),
    )
