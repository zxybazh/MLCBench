from typing import List, Dict, Union, Tuple, Callable
from pathlib import Path
import cloudpickle

import tvm

IMPORTS = """import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench
"""

MAIN = """if __name__ == "__main__":
    bench = MLCBench()
    for _ in range(SAMPLE_NUMBER):
        input_infos, median, std = bench.benchmark(
            main, input_shape_gen_func, "llvm -num-cores=4", tvm.cpu()
        )
        bench.record(
            RELAX_FUNC_NAME,
            PRIM_FUNC_NAME,
            input_infos,
            median,
            std,
            WEIGHT,
            WEIGHT*median
        )
    bench.show()
"""


def counted(func: Callable):
    def wrapped(*args, **kwargs):
        wrapped.count += 1
        return func(*args, **kwargs)

    wrapped.count = -1
    return wrapped


def extract_shape(
    arg: Union[tuple, list, tvm.relax.Tuple, tvm.relax.ShapeStructInfo]
) -> List[tvm.relax.ShapeStructInfo]:
    if isinstance(arg, (tuple, list, tvm.relax.Tuple)):
        results = []
        for sub_arg in arg:
            results.extend(extract_shape(sub_arg))
        return results
    else:
        return [arg.struct_info]


def prim_func_usage_gen(
    mod: tvm.ir.IRModule,
) -> Tuple[tvm.ir.GlobalVar, tvm.ir.GlobalVar, List[tvm.relax.ShapeStructInfo]]:
    for gv, func in mod.functions.items():
        if isinstance(func, tvm.relax.Function):
            for block in func.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding.value, tvm.relax.expr.Call):
                        raw_args = binding.value.args
                        functor = raw_args[0]
                        if isinstance(functor, tvm.ir.GlobalVar):
                            if isinstance(mod.functions[functor], tvm.tir.PrimFunc):
                                args = extract_shape(raw_args[1:]) + extract_shape(
                                    binding.value
                                )
                                yield gv, functor, args


def extract_dynamic_var(
    func_dict: Dict,
) -> Dict[tvm.ir.GlobalVar, Dict[tvm.relax.expr.Call, str]]:
    dynamic_var_dict = {}
    for gv in func_dict:
        dynamic_var_dict[gv] = {}
        for functor in func_dict[gv]:
            for arg_list, _ in func_dict[gv][functor]:
                for arg in arg_list:
                    if isinstance(arg, tvm.relax.TensorStructInfo):
                        for v in arg.shape.values:
                            if isinstance(v, tvm.tir.Var):
                                dynamic_var_dict[gv][v] = v.dtype
                    elif isinstance(arg, tvm.relax.ShapeStructInfo):
                        for v in arg.values:
                            if isinstance(v, tvm.tir.Var):
                                dynamic_var_dict[gv][v] = v.dtype
                    else:
                        raise NotImplementedError
    return dynamic_var_dict


def extract_func_info(
    mod: tvm.ir.IRModule,
) -> Dict[tvm.ir.GlobalVar, Dict[tvm.ir.GlobalVar, List[Tuple[List, int]]]]:
    """Extract function information from a relax module."""

    def update_records(records, new_args):
        for i, (args, count) in enumerate(records):
            if new_args == args:
                records[i] = (args, count + 1)
                return
        records.append((new_args, 1))

    relax_func_dict = {}
    for gv, functor, args in prim_func_usage_gen(mod):
        if isinstance(functor, tvm.ir.GlobalVar):
            if not gv in relax_func_dict:
                relax_func_dict[gv] = {}
            if not functor in relax_func_dict[gv]:
                relax_func_dict[gv][functor] = []
            update_records(relax_func_dict[gv][functor], args)
    return relax_func_dict, extract_dynamic_var(relax_func_dict)


def get_shape_gen_func(func_args: List[Tuple[tvm.relax.expr.Call, str]]):
    def produce_shape(shape: tvm.relax.expr.ShapeExpr, count: int) -> Tuple[int]:
        produced = []
        for var in shape:
            if isinstance(var, tvm.tir.IntImm):
                produced.append(var.value)
            else:
                produced.append(2 ** (count % 13))
        return tuple(produced)

    @counted
    def input_shape_gen_func():
        results = []
        for arg in func_args:
            if isinstance(arg, tvm.relax.ShapeStructInfo):
                pass
            elif isinstance(arg, tvm.relax.struct_info.TensorStructInfo):
                results.append(
                    (
                        produce_shape(arg.shape, input_shape_gen_func.count),
                        arg.dtype,
                    )
                )
            else:
                for sub_arg in arg.fields:
                    results.append(
                        (
                            produce_shape(sub_arg.shape, input_shape_gen_func.count),
                            sub_arg.dtype,
                        )
                    )

        # work around wrong input order for integer input
        for arg in func_args:
            if isinstance(arg, tvm.relax.ShapeStructInfo):
                results.append(
                    (
                        (),
                        "int64",
                    )
                )
        return results

    return input_shape_gen_func


def extract_prim_func(
    model_name: str,
    relax_func_name: str,
    prim_func_name: str,
    func: tvm.tir.PrimFunc,
    func_args: List[Tuple[tvm.relax.expr.Call, str]],
    weight: int,
    file_path: str,
    category: int = -1,
    sample_number: int = 5,
):
    file = open(file_path, "w")
    print(IMPORTS, file=file)
    print(f'MODEL_NAME = "{model_name}"', file=file)
    print(f'RELAX_FUNC_NAME = "{relax_func_name}"', file=file)
    print(f'PRIM_FUNC_NAME = "{prim_func_name}"', file=file)
    print(f"FUNC_HASH = {tvm.ir.structural_hash(func)}", file=file)
    print(f"WEIGHT = {weight}", file=file)
    print(f"CAT = {category}", file=file)
    print(f"SAMPLE_NUMBER = {sample_number}\n", file=file)
    print(
        f"input_shape_gen_func = pickle.loads({cloudpickle.dumps(get_shape_gen_func(func_args))})",
        file=file,
    )
    print(func.script() + "\n", file=file)
    print(MAIN, file=file)


def get_func_name_from_gv(gv: tvm.ir.GlobalVar) -> str:
    return gv.astext().split("@")[1] if "@" in gv.astext() else gv.astext()


def extract_from_relax(mod: tvm.ir.IRModule, model_name: str, file_path: str):
    relax_funcs = extract_func_info(mod)
    Path(file_path).mkdir(parents=True, exist_ok=True)
    for relax_func_gv in relax_funcs:
        relax_func_name = get_func_name_from_gv(relax_func_gv)
        for prim_func_gv in relax_funcs[relax_func_gv]:
            prim_func_name = get_func_name_from_gv(prim_func_gv)
            for func_args, weight in relax_funcs[relax_func_gv][prim_func_gv]:
                extract_prim_func(
                    model_name=model_name,
                    relax_func_name=relax_func_name,
                    prim_func_name=prim_func_name,
                    func=mod[prim_func_gv],
                    func_args=func_args,
                    weight=weight,
                    file_path=f"{file_path}/{relax_func_name}_{prim_func_name}.py",
                )


if __name__ == "__main__":
    raise NotImplementedError
