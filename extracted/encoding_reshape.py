import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "encoding"
PRIM_FUNC_NAME = "reshape"
FUNC_HASH = -5351629835429146286
WEIGHT = 1
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95\xce\n\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xd2\x05\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "int32", \n        "ndim": "2", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "11", \n        "span": "0", \n        "struct_info_": "8", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "6", \n        "span": "0", \n        "type_annotation": "7"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "2", \n        "span": "0", \n        "values": "9"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [10, 5]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "2", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nXz\x04\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "int32", \n        "ndim": "1", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "8", \n        "span": "0", \n        "struct_info_": "7", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4]\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "5", \n        "span": "0", \n        "type_annotation": "6"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "1", \n        "span": "0", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "1", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x1d\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01n\x94\x8c\x05int64\x94\x8c\x01m\x94\x8c\x05int64\x94u.')

# from tvm.script import tir as T

@T.prim_func
def main(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n), "int32")
    T_reshape = T.match_buffer(var_T_reshape, (n,), "int32")
    # with T.block("root"):
    for ax0 in range(n):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(n, ax0)
            T.reads(A[T.int64(0), v_ax0 % n])
            T.writes(T_reshape[v_ax0])
            T_reshape[v_ax0] = A[T.int64(0), v_ax0 % n]

if __name__ == "__main__":
    bench = MLCBench()
    target = tvm.target.Target("llvm -keys=cpu -num-cores=4")
    dev = tvm.cpu()
    print("Input args:", INPUT_ARGS)
    for _ in range(SAMPLE_NUMBER):
        dym_var_sample = DYM_VAR_SAMPLE_FUNC(DYM_VAR_DICT)
        input_infos, median, std = bench.benchmark(
            main,
            INPUT_ARGS,
            dym_var_sample=dym_var_sample,
            target=target,
            dev=dev,
        )
        bench.record(
            {
                "RelaxFunc": RELAX_FUNC_NAME,
                "PrimFunc": PRIM_FUNC_NAME,
                "InputInfo": ", ".join(
                    [f"{k} = {v}" for k, v in dym_var_sample.items()]
                ),
                "Time(us)": median*1e6,
                "Std(us)": std*1e6,
                "Weight": WEIGHT,
                "WxTime(ms)": WEIGHT*median*1e3,
            }
        )
    bench.show()

