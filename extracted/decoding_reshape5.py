import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "decoding"
PRIM_FUNC_NAME = "reshape5"
FUNC_HASH = -1168775914808931150
WEIGHT = 1
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95\xab\n\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xa2\x05\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "int32", \n        "ndim": "2", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "10", \n        "span": "0", \n        "struct_info_": "6", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "2", \n        "span": "0", \n        "values": "7"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [8, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "2", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\x87\x04\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "int32", \n        "ndim": "1", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "8", \n        "span": "0", \n        "struct_info_": "5", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "1", \n        "span": "0", \n        "values": "6"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [7]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "1", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x10\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x01n\x94\x8c\x05int64\x94s.')

# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
    T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0 in range(T.int64(1)):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(T.int64(1), ax0)
            T.reads(A[T.int64(0), T.int64(0)])
            T.writes(T_reshape[v_ax0])
            T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]

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

