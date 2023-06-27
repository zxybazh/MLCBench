import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "encoding"
PRIM_FUNC_NAME = "fused_min_max_triu_te_broadcast_to"
FUNC_HASH = 2282863803154781898
WEIGHT = 1
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95#\n\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x0fShapeStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\x97\x02\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "1", \n        "span": "0", \n        "values": "2"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [3]\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "4", \n        "span": "0", \n        "type_annotation": "5"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x04\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94h\nX\xf6\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "13", \n        "span": "0", \n        "struct_info_": "9", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6, 6]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "7", \n        "span": "0", \n        "type_annotation": "8"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "10"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [11, 12, 6, 6]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x1d\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01n\x94\x8c\x05int64\x94\x8c\x01m\x94\x8c\x05int64\x94u.')

# from tvm.script import tir as T

@T.prim_func
def main(p_output0: T.handle, n: T.int64):
    T.func_attr({"tir.noalias": T.bool(True)})
    var_T_broadcast_to_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), n, n), "float16")
    # with T.block("root"):
    var_make_diag_mask_te_intermediate = T.alloc_buffer((n, n), "float16")
    for i, j in T.grid(n, n):
        with T.block("make_diag_mask_te"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads()
            T.writes(var_make_diag_mask_te_intermediate[v_i, v_j])
            var_make_diag_mask_te_intermediate[v_i, v_j] = T.Select(v_i < v_j, T.float16(-65504), T.float16(0))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), n, n):
        with T.block("T_broadcast_to"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_make_diag_mask_te_intermediate[v_ax2, v_ax3])
            T.writes(var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_make_diag_mask_te_intermediate[v_ax2, v_ax3]

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

