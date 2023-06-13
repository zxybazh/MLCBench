from typing import Callable, Union, Tuple, List

import tvm
from tvm.ir import IRModule
from tvm.tir import PrimFunc
from tvm.meta_schedule.testing.tune_utils import generate_input_data

import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


class MLCBench:
    """Benchmarking PrimFuncs or IRModules with Dynamic Input Shapes."""

    df: pd.DataFrame = pd.DataFrame(
        columns=["Name", "Input Info", "Time(ms)", "Std(ms)"]
    )

    @staticmethod
    def benchmark(
        mod_or_func: Union[PrimFunc, IRModule],
        input_shape_gen_func: Callable[[], Tuple[int]],
        target: str = "nvidia/nvidia-a10g",
        dev: tvm.runtime.Device = tvm.cuda(),
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
        for input_shape, input_dtype in input_infos:
            input_tensors.append(
                tvm.nd.array(generate_input_data(input_shape, input_dtype), device=dev)
                if input_shape != ()
                else 1  # random integer
            )
        rt_mod = tvm.build(mod, target=target)
        result = rt_mod.time_evaluator(global_symbol, dev=dev, number=10, repeat=10)(
            *input_tensors
        )
        return input_infos, result.median, result.std

    @staticmethod
    def record(
        func_name: str,
        input_infos: List[Tuple[Tuple[int], str]],
        median: float,
        std: float,
    ):
        MLCBench.df = MLCBench.df._append(
            {
                "Name": func_name,
                "Input Info": input_infos,
                "Time(ms)": median * 1000,
                "Std(ms)": std * 1000,
            },
            ignore_index=True,
        )

    @staticmethod
    def show():
        print(MLCBench.df)


if __name__ == "__main__":
    raise NotImplementedError
