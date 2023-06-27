from typing import List, Dict, Union, Tuple, Callable
from pathlib import Path
import cloudpickle

import tvm

SKETCH = """import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "{model_name}"
RELAX_FUNC_NAME = "{relax_func_name}"
PRIM_FUNC_NAME = "{prim_func_name}"
FUNC_HASH = {func_hash}
WEIGHT = {weight}
CAT = {category}
SAMPLE_NUMBER = {sample_number}

INPUT_SHAPE_GEN_FUNC = pickle.loads({func_dump})

{func_script}

if __name__ == "__main__":
    bench = MLCBench()
    target = tvm.target.Target({target})
    dev = {dev}
    for _ in range(SAMPLE_NUMBER):
        input_infos, median, std = bench.benchmark(
            main, args, dym_var_sample=, target=target, dev=dev, number=10, repeat=10
        )
        bench.record(
        **{
            "PrimFunc": PRIM_FUNC_NAME,
            "InputInfo": input_infos,
            "Time(us)": median*1e6,
            "Std(us)": std*1e6,
            "Weight": weight,
            "WxTime(ms)": weight*median*1e3,
        })
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
    target: Union[str, tvm.target.Target] = "llvm -num-cores=4",
):
    target = tvm.target.Target(target)

    if target.kind.name == "cuda":
        dev = "tvm.cuda()"
    elif target.kind.name == "llvm":
        dev = "tvm.cpu()"
    else:
        raise NotImplementedError("Only support cuda and llvm target.")

    file = open(file_path, "w")

    print(
        SKETCH.format(
            **{
                "model_name": model_name,
                "relax_func_name": relax_func_name,
                "prim_func_name": prim_func_name,
                "func_hash": tvm.ir.structural_hash(func),
                "weight": weight,
                "category": category,
                "sample_number": sample_number,
                "func_dump": cloudpickle.dumps(get_shape_gen_func(func_args)),
                "func_script": func.script(),
                "target": tvm.target.Target(target),
                "dev": dev,
            }
        ),
        file=file,
    )


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
