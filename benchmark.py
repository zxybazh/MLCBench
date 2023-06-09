import tvm
from typing import Callable, Union
from tvm.ir import IRModule
from tvm.tir import PrimFunc, Schedule
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.meta_schedule.testing.tune_utils import generate_input_data

import pandas as pd

# fmt: off
@T.prim_func
def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), n))
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm_1[v_bsz, v_i, v_k])
            rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
# fmt: on


def counted(func: Callable):
    def wrapped(*args, **kwargs):
        wrapped.count += 1
        return func(*args, **kwargs)

    wrapped.count = 0
    return wrapped


@counted
def rms_norm_input_shape_gen_func():
    factor = 2 ** (rms_norm_input_shape_gen_func.count % 13)
    return [
        ((1, factor, 4096), "float16"),
        ((4096), "float16"),
        ((1, factor, 4096), "float16"),
    ]


def benchmark(
    mod_or_func: Union[PrimFunc, IRModule],
    input_shape_gen_func: Callable,
    target: str = "nvidia/nvidia-a10g",
    dev: tvm.runtime.Device = tvm.cuda(),
):
    if isinstance(mod_or_func, PrimFunc):
        mod = IRModule.from_expr(mod_or_func)
    else:
        mod = mod_or_func
    if isinstance(target, str):
        target = tvm.target.Target(target)
    input_infos = input_shape_gen_func()
    input_tensors = []
    for input_shape, input_dtype in input_infos:
        input_tensors.append(
            tvm.nd.array(generate_input_data(input_shape, input_dtype), device=dev)
        )
    rt_mod = tvm.build(mod, target=target)
    result = rt_mod.time_evaluator("main", dev=dev, number=10, repeat=10)(
        *input_tensors
    )
    return input_infos, result.median, result.std


if __name__ == "__main__":
    df = pd.DataFrame(columns=["Name", "Input Info", "Time(ms)", "Std(ms)"])
    for _ in range(10):
        input_infos, median, std = benchmark(
            rms_norm, rms_norm_input_shape_gen_func, "llvm -num-cores=4", tvm.cpu()
        )
        df = df._append(
            {
                "Name": "rms_norm",
                "Input Info": input_infos,
                "Time(ms)": median * 1000,
                "Std(ms)": std * 1000,
            },
            ignore_index=True,
        )
    print(df)
