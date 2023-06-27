import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "encoding"
PRIM_FUNC_NAME = "fused_silu_multiply"
FUNC_HASH = 4940283264744739424
WEIGHT = 32
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95~\x15\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xf8\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "3", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "13", \n        "span": "0", \n        "struct_info_": "9", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 8]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "6", \n        "span": "0", \n        "type_annotation": "7"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "11008"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "3", \n        "span": "0", \n        "values": "10"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [11, 5, 12]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "11008"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "3", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\xf8\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "3", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "13", \n        "span": "0", \n        "struct_info_": "9", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 8]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "6", \n        "span": "0", \n        "type_annotation": "7"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "11008"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "3", \n        "span": "0", \n        "values": "10"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [11, 5, 12]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "11008"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "3", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\xf8\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "3", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "13", \n        "span": "0", \n        "struct_info_": "9", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 8]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "6", \n        "span": "0", \n        "type_annotation": "7"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "11008"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "3", \n        "span": "0", \n        "values": "10"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [11, 5, 12]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "11008"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "3", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x1d\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01n\x94\x8c\x05int64\x94\x8c\x01m\x94\x8c\x05int64\x94u.')

# from tvm.script import tir as T

@T.prim_func
def main(p_lv4: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv4 = T.match_buffer(p_lv4, (T.int64(1), n, T.int64(11008)), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(lv4[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(lv4[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv4[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = lv4[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2], lv5[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] * lv5[v_ax0, v_ax1, v_ax2]

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

