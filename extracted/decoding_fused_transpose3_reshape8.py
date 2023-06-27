import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "decoding"
PRIM_FUNC_NAME = "fused_transpose3_reshape8"
FUNC_HASH = 3103818021196041790
WEIGHT = 32
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95*\x0f\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xe2\x07\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "14", \n        "span": "0", \n        "struct_info_": "8", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6, 7]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "9"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [10, 11, 12, 13]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\xc6\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "3", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "12", \n        "span": "0", \n        "struct_info_": "7", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "3", \n        "span": "0", \n        "values": "8"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [9, 10, 11]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "3", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x10\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x01n\x94\x8c\x05int64\x94s.')

# from tvm.script import tir as T

@T.prim_func
def main(lv1517: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16")
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv1517[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1517[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)]

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

