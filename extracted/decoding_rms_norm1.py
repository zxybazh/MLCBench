import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "decoding"
PRIM_FUNC_NAME = "rms_norm1"
FUNC_HASH = 2292435488911535817
WEIGHT = 65
CAT = -1
SAMPLE_NUMBER = 5

INPUT_ARGS = pickle.loads(b'\x80\x05\x95\xb1\x12\x00\x00\x00\x00\x00\x00]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x10TensorStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xc6\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "3", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "12", \n        "span": "0", \n        "struct_info_": "7", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "3", \n        "span": "0", \n        "values": "8"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [9, 10, 11]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "3", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\x8f\x04\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "1", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "8", \n        "span": "0", \n        "struct_info_": "5", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "1", \n        "span": "0", \n        "values": "6"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [7]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "1", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x03h\x06\x85\x94R\x94}\x94h\nX\xc6\x06\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "3", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "12", \n        "span": "0", \n        "struct_info_": "7", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "3", \n        "span": "0", \n        "values": "8"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [9, 10, 11]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "4096"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "3", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe.')
DYM_VAR_SAMPLE_FUNC = pickle.loads(b'\x80\x05\x958\x00\x00\x00\x00\x00\x00\x00\x8c\x14mlc_bench.extraction\x94\x8c\x1bdefault_dym_var_sample_func\x94\x93\x94.')
DYM_VAR_DICT = pickle.loads(b'\x80\x05\x95\x10\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x01n\x94\x8c\x05int64\x94s.')

# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), B: T.Buffer((T.int64(4096),), "float16"), rms_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), T.int64(1)))
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm[v_bsz, v_i, v_k])
            rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

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

