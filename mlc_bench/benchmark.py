from typing import Callable, Union, Tuple, List, Dict, Any
import pandas as pd

import tvm
from tvm.ir import IRModule
from tvm.tir import PrimFunc
from tvm.meta_schedule.testing.tune_utils import generate_input_data

from .extraction import (
    extract_func_info,
    get_func_name_from_gv,
    default_dym_var_sample_func,
)


def dataframe_sort_by(df: pd.DataFrame, sort_by: str):
    """Sort a dataframe by a column and reset index."""
    return df.sort_values(sort_by, ascending=False).reset_index().drop("index", axis=1)


def val_value_str(sample: Dict[tvm.relax.expr.Call, int]) -> str:
    """Convert a variable value sample to a string."""
    return ", ".join([f"{k}={v}" for k, v in sample.items()])


def populuate_input_shape(
    input_infos: List[Tuple[tvm.relax.TensorStructInfo]],
    dym_var_sample: Dict[str, int],
) -> List[Tuple[Tuple[int], str]]:
    """
    Populate input shapes with dynamic shape variable samples.

    [..., Shape(1, n, 128) with dtype=int32, ...] ->
    [..., ((1, 64, 128), "int32"), ...] if n=64

    """
    results = []
    for input_info in input_infos:
        shape = []
        if isinstance(input_info, tvm.relax.struct_info.ShapeStructInfo):
            # scalar input
            results.append((dym_var_sample[str(input_info.values[0])], "scalar"))
        else:
            for dim in input_info.shape:
                if isinstance(dim, tvm.tir.IntImm):
                    shape.append(dim.value)
                else:
                    shape.append(dym_var_sample[str(dim)])
            results.append((tuple(shape), input_info.dtype))
    return results


class MLCBench:
    """Benchmarking PrimFuncs or IRModules with Dynamic Input Shapes."""

    df: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def benchmark(
        mod_or_func: Union[PrimFunc, IRModule],
        args: List[tvm.relax.TensorStructInfo],
        dym_var_sample: Dict[Union[tvm.relax.expr.Call, str], int],
        target: Union[str, tvm.target.Target] = "llvm -num-cores=4",
        dev: tvm.runtime.Device = tvm.cpu(),
        number=10,
        repeat=10,
    ) -> Tuple[List[Tuple[Tuple[int], str]], float, float]:
        """Benchmark a PrimFunc or IRModule with dynamic input shapes."""
        # produce IRModule and function name
        if isinstance(mod_or_func, PrimFunc):
            mod = IRModule.from_expr(mod_or_func.with_attr("global_symbol", "main"))
            func_name = "main"
        else:
            mod = mod_or_func
            # assume only one global function
            (func_name,) = mod.get_global_vars()
        # produce target
        target = tvm.target.Target(target)
        # populate input shapes
        input_infos = populuate_input_shape(args, dym_var_sample)
        # generate input tensors, including scalars
        # scalars are appended to the end of the list due to parsing order
        input_tensors = []
        scalar_input_tensors = []
        for input_shape, input_dtype in input_infos:
            if input_dtype == "scalar":
                # special case like [n], generate int value
                assert isinstance(input_shape, int)
                scalar_input_tensors.append(input_shape)
            else:
                # normal case like [1, n, 128], generate random tensor
                input_tensors.append(
                    tvm.nd.array(
                        generate_input_data(input_shape, input_dtype), device=dev
                    )
                )
        # append scalar input tensors
        input_tensors.extend(scalar_input_tensors)
        # build locally
        rt_mod = tvm.build(mod, target=target)
        # benchmark locally
        result = rt_mod.time_evaluator(
            func_name, dev=dev, number=number, repeat=repeat
        )(*input_tensors)
        # return input infos, median, std
        return input_infos, result.median, result.std

    @staticmethod
    def record(
        kwargs: Dict[str, Any],
    ):
        MLCBench.df = MLCBench.df._append(
            kwargs,
            ignore_index=True,
        )

    @staticmethod
    def show(sort_by: str = "WxTime(ms)"):
        print(dataframe_sort_by(MLCBench.df, sort_by))
        print("\n")

    @staticmethod
    def clear():
        MLCBench.df = pd.DataFrame()

    @staticmethod
    def benchmark_relax_func(
        mod: tvm.ir.IRModule,
        relax_func: Union[tvm.ir.GlobalVar, str] = None,
        sample_number: int = 2,
        dym_var_sample_func: Callable[
            [Dict[str, str]],
            Dict[str, int],
        ] = default_dym_var_sample_func,
        target: Union[str, tvm.target.Target] = "llvm -num-cores=4",
        dev: tvm.runtime.Device = tvm.cpu(),
        number=10,
        repeat=10,
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
                            args,
                            dym_var_sample,
                            target=target,
                            dev=dev,
                            number=number,
                            repeat=repeat,
                        )
                        MLCBench.record(
                            {
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
