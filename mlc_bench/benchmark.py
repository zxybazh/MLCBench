from typing import Callable, Union, Tuple, List, Dict, Any

import tvm
from tvm.ir import IRModule
from tvm.tir import PrimFunc
from tvm.meta_schedule.testing.tune_utils import generate_input_data

import pandas as pd
import random

from .extraction import extract_func_info, get_func_name_from_gv

COLUMNS = [
    "Time(us)",
    #    "Std(us)",
    "Weight",
    "WxTime(ms)",
]


def dedup(func: Callable):
    def wrapped(*args, **kwargs):
        factor = func(*args, **kwargs)
        fail_count = 0
        while factor in wrapped.used and fail_count < 10:
            factor = func(*args, **kwargs)
            fail_count += 1
        wrapped.used.append(factor)
        return factor

    wrapped.used = []
    return wrapped


@dedup
def default_dym_var_sample_func(
    dym_var_dict: Dict[tvm.relax.expr.Call, str]
) -> Dict[tvm.relax.expr.Call, int]:
    results = {}
    for var in dym_var_dict:
        if dym_var_dict[var] == "int32" or "int64":
            results[var] = 2 ** random.randint(2, 10)
        else:
            raise TypeError(
                "Unsupported dynamic shape variable type: " + dym_var_dict[var]
            )
    return results


def val_value_str(sample: Dict[tvm.relax.expr.Call, int]) -> str:
    return ", ".join([f"{k}={v}" for k, v in sample.items()])


def populuate_input_shape(
    input_infos: List[Tuple[tvm.relax.TensorStructInfo]],
    dym_var_sample: Dict[tvm.relax.expr.Call, int],
) -> List[Tuple[Tuple[int], str]]:
    results = []
    for input_info in input_infos:
        shape = []
        if isinstance(input_info, tvm.relax.struct_info.ShapeStructInfo):
            # scalar input
            results.append((dym_var_sample[input_info.values[0]], "scalar"))
        else:
            for dim in input_info.shape:
                if isinstance(dim, tvm.tir.IntImm):
                    shape.append(dim.value)
                else:
                    shape.append(dym_var_sample[dim])
            results.append((tuple(shape), input_info.dtype))
    return results


class MLCBench:
    """Benchmarking PrimFuncs or IRModules with Dynamic Input Shapes."""

    df: pd.DataFrame = pd.DataFrame(columns=COLUMNS)

    @staticmethod
    def benchmark(
        mod_or_func: Union[PrimFunc, IRModule],
        input_shape_gen_func: Callable[[], Tuple[int]],
        target: str = "nvidia/nvidia-a10g",
        dev: tvm.runtime.Device = tvm.cuda(),
        number=1,
        repeat=1,
    ) -> Tuple[List[Tuple[Tuple[int], str]], float, float]:
        if isinstance(mod_or_func, PrimFunc):
            func = mod_or_func
            func = func.with_attr("global_symbol", "main")
            mod = IRModule.from_expr(func)
            global_symbol = "main"
        else:
            mod = mod_or_func
            global_symbol = mod.get_global_vars()[0]
        if isinstance(target, str):
            target = tvm.target.Target(target)
        input_infos = input_shape_gen_func()
        input_tensors = []
        scalar_input_tensors = []
        for input_shape, input_dtype in input_infos:
            if input_dtype == "scalar":
                assert isinstance(input_shape, int)
                scalar_input_tensors.append(input_shape)
            else:
                input_tensors.append(
                    tvm.nd.array(
                        generate_input_data(input_shape, input_dtype), device=dev
                    )
                )
        input_tensors.extend(scalar_input_tensors)
        rt_mod = tvm.build(mod, target=target)
        result = rt_mod.time_evaluator(
            global_symbol, dev=dev, number=number, repeat=repeat
        )(*input_tensors)
        return input_infos, result.median, result.std

    @staticmethod
    def record(
        **kwargs,
    ):
        MLCBench.df = MLCBench.df._append(
            kwargs,
            ignore_index=True,
        )

    @staticmethod
    def show():
        print(MLCBench.df)
        print("\n")

    @staticmethod
    def clear():
        MLCBench.df = pd.DataFrame(columns=COLUMNS)

    @staticmethod
    def sample_input_shape(
        prim_funcs: Dict[tvm.ir.GlobalVar, List[Tuple[tvm.relax.expr.Call, str]]],
        get_var_val_func: Callable[[tvm.relax.expr], int] = lambda x: random.randint(
            1, 100
        ),
    ) -> Dict[tvm.ir.GlobalVar, List[Tuple[Tuple[int], str]]]:
        rvs = {}
        results = {}
        for prim_func_gv in prim_funcs.items():
            for func_args, _ in prim_funcs[prim_func_gv]:
                for arg in func_args:
                    if isinstance(arg, tvm.relax.ShapeStructInfo):
                        for var in arg:
                            rvs[var] = 0
                    elif isinstance(arg, tvm.relax.struct_info.TensorStructInfo):
                        for var in arg.shape:
                            if not isinstance(var, tvm.tir.IntImm):
                                rvs[var] = 0
                    else:
                        for sub_arg in arg.fields:
                            for var in sub_arg.shape:
                                if not isinstance(var, tvm.tir.IntImm):
                                    rvs[var] = 0
        for var in rvs:
            rvs[var] = get_var_val_func()
        for prim_func_gv in prim_funcs.items():
            results[prim_func_gv] = []
            for func_args, weight in prim_funcs[prim_func_gv]:
                for arg in func_args:
                    if isinstance(arg, tvm.relax.ShapeStructInfo):
                        shape = []
                        for var in arg:
                            shape.append(rvs[var])
                        results[prim_func_gv].appeend((tuple(shape), weight))
                    elif isinstance(arg, tvm.relax.struct_info.TensorStructInfo):
                        shape = []
                        for var in arg.shape:
                            shape.append(rvs[var])
                        results[prim_func_gv].appeend((tuple(shape), weight))
                    else:
                        for sub_arg in arg.fields:
                            shape = []
                            for var in sub_arg.shape:
                                if not isinstance(var, tvm.tir.IntImm):
                                    shape.append(rvs[var])

    @staticmethod
    def benchmark_relax_func(
        mod: tvm.ir.IRModule,
        relax_func: Union[tvm.ir.GlobalVar, str] = None,
        sample_number: int = 2,
        target: str = "nvidia/nvidia-a10g",
        dev: tvm.runtime.Device = tvm.cuda(),
        dym_var_sample_func: Callable[
            [Dict[tvm.relax.expr.Call, str]],
            Dict[tvm.relax.expr.Call, int],
        ] = default_dym_var_sample_func,
    ):
        relax_funcs, dynamic_var_dict = extract_func_info(mod)

        if relax_func is None:
            for gv in relax_funcs:
                MLCBench.benchmark_relax_func(
                    mod,
                    gv,
                    sample_number,
                    target,
                    dev,
                    dym_var_sample_func,
                )
        else:
            if isinstance(relax_func, str):
                for gv in relax_funcs:
                    if get_func_name_from_gv(gv) == relax_func:
                        relax_func = gv
                        break
                if not isinstance(relax_func, tvm.ir.GlobalVar):
                    raise ValueError(
                        f"Cannot find relax function with name {relax_func}"
                    )
            for _ in range(sample_number):
                MLCBench.clear()
                dym_var_sample = dym_var_sample_func(dynamic_var_dict[relax_func])
                for functor in relax_funcs[relax_func]:
                    for args, weight in relax_funcs[relax_func][functor]:
                        input_infos, median, std = MLCBench.benchmark(
                            mod[functor],
                            input_shape_gen_func=lambda: populuate_input_shape(
                                args, dym_var_sample
                            ),
                            target=target,
                            dev=dev,
                        )
                        MLCBench.record(
                            **{
                                f"PrimFuncs in {get_func_name_from_gv(relax_func)}": get_func_name_from_gv(
                                    functor
                                ),
                                f"InputInfo({val_value_str(dym_var_sample)})": ", ".join(
                                    [str(w) for w in args]
                                ),
                                "Time(us)": median * 1e6,
                                # "Std(us)": std * 1e6,
                                "Weight": weight,
                                "WxTime(ms)": median * weight * 1e3,
                            }
                        )
                MLCBench.show()


if __name__ == "__main__":
    raise NotImplementedError
