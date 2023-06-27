import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "encoding"
PRIM_FUNC_NAME = "matmul"
FUNC_HASH = -707577035993519378
WEIGHT = 32
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95\xf8\x18\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X>\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "16", \n        "span": "0", \n        "struct_info_": "12", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "7", \n        "span": "0", \n        "type_annotation": "8"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "10", \n        "span": "0", \n        "type_annotation": "11"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "m"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "13"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [14, 15, 6, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\x12\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "15", \n        "span": "0", \n        "struct_info_": "10", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "7", \n        "span": "0", \n        "type_annotation": "8"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "m"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "11"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [12, 13, 6, 14]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\x12\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "15", \n        "span": "0", \n        "struct_info_": "10", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "7", \n        "span": "0", \n        "type_annotation": "8"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "11"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [12, 13, 6, 14]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x1d\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01n\x94\x8c\x05int64\x94\x8c\x01m\x94\x8c\x05int64\x94u.')

# from tvm.script import tir as T

@T.prim_func
def main(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    matmul_1 = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), m):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul_1[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul_1[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul_1[v_i0, v_i1, v_i2, v_i3] = matmul_1[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

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

