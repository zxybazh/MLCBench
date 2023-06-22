import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "vicuna_v1_7b_fp_16"
RELAX_FUNC_NAME = "encoding"
PRIM_FUNC_NAME = "transpose1"
FUNC_HASH = -8427332538457172017
WEIGHT = 32
CAT = -1
SAMPLE_NUMBER = 5

input_shape_gen_func = pickle.loads(b'\x80\x05\x95f\x18\x00\x00\x00\x00\x00\x00\x8c\x17cloudpickle.cloudpickle\x94\x8c\x0e_make_function\x94\x93\x94(h\x00\x8c\r_builtin_type\x94\x93\x94\x8c\x08CodeType\x94\x85\x94R\x94(K\x00K\x00K\x00K\x02K\x04K\x1fC\x1c\x88\x01\x04\x00j\x00d\x017\x00\x02\x00_\x00\x88\x00|\x00i\x00|\x01\xa4\x01\x8e\x01S\x00\x94NK\x01\x86\x94\x8c\x05count\x94\x85\x94\x8c\x04args\x94\x8c\x06kwargs\x94\x86\x94\x8c./home/ubuntu/mlc_bench/mlc_bench/extraction.py\x94\x8c\x07wrapped\x94KDC\x04\x0e\x01\x0e\x01\x94\x8c\x04func\x94h\x10\x86\x94)t\x94R\x94}\x94(\x8c\x0b__package__\x94\x8c\tmlc_bench\x94\x8c\x08__name__\x94\x8c\x14mlc_bench.extraction\x94\x8c\x08__file__\x94\x8c./home/ubuntu/mlc_bench/mlc_bench/extraction.py\x94uNNh\x00\x8c\x10_make_empty_cell\x94\x93\x94)R\x94h\x1e)R\x94\x86\x94t\x94R\x94\x8c\x1ccloudpickle.cloudpickle_fast\x94\x8c\x12_function_setstate\x94\x93\x94h#}\x94h\nJ\xff\xff\xff\xffs}\x94(h\x19h\x10\x8c\x0c__qualname__\x94\x8c4get_shape_gen_func.<locals>.counted.<locals>.wrapped\x94\x8c\x0f__annotations__\x94}\x94\x8c\x0e__kwdefaults__\x94N\x8c\x0c__defaults__\x94N\x8c\n__module__\x94h\x1a\x8c\x07__doc__\x94N\x8c\x0b__closure__\x94h\x00\x8c\n_make_cell\x94\x93\x94h\x02(h\x07(K\x00K\x00K\x00K\x03K\x07K\x13C\x96g\x00}\x00\x88\x00D\x00]3}\x01t\x00|\x01t\x01j\x02j\x03\x83\x02r\x0eq\x04t\x00|\x01t\x01j\x02j\x04j\x05\x83\x02r$|\x00\xa0\x06\x88\x02|\x01j\x07\x88\x01j\x08\x83\x02|\x01j\tf\x02\xa1\x01\x01\x00q\x04|\x01j\nD\x00]\x0f}\x02|\x00\xa0\x06\x88\x02|\x02j\x07\x88\x01j\x08\x83\x02|\x02j\tf\x02\xa1\x01\x01\x00q\'q\x04\x88\x00D\x00]\x0e}\x01t\x00|\x01t\x01j\x02j\x03\x83\x02rH|\x00\xa0\x06d\x01\xa1\x01\x01\x00q:|\x00S\x00\x94N)\x8c\x05int64\x94\x86\x94\x86\x94(\x8c\nisinstance\x94\x8c\x03tvm\x94\x8c\x05relax\x94\x8c\x0fShapeStructInfo\x94\x8c\x0bstruct_info\x94\x8c\x10TensorStructInfo\x94\x8c\x06append\x94\x8c\x05shape\x94h\n\x8c\x05dtype\x94\x8c\x06fields\x94t\x94\x8c\x07results\x94\x8c\x03arg\x94\x8c\x07sub_arg\x94\x87\x94h\x0f\x8c\x14input_shape_gen_func\x94KTC0\x04\x02\x08\x01\x0e\x01\x02\x01\x10\x01\x04\x01\x0c\x02\x04\x01\x02\xfe\x06\xff\n\x07\x04\x01\x0c\x02\x04\x01\x02\xfe\x06\xff\x02\xff\x08\t\x0e\x01\x04\x01\x02\x01\x04\xff\x02\x80\x04\x06\x94\x8c\tfunc_args\x94hG\x8c\rproduce_shape\x94\x87\x94)t\x94R\x94h\x16NNh\x1e)R\x94h\x1e)R\x94h\x1e)R\x94\x87\x94t\x94R\x94h&hS}\x94}\x94(h\x19hGh)\x8c0get_shape_gen_func.<locals>.input_shape_gen_func\x94h+}\x94h-Nh.Nh/h\x1ah0Nh1h3]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x0fTupleStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xcf\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TupleStructInfo", \n      "attrs": {\n        "fields": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [3]\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "4", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "17", \n        "span": "0", \n        "struct_info_": "12", \n        "values": "5"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [6, 7, 8, 11]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "9", \n        "span": "0", \n        "type_annotation": "10"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "13"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [14, 15, 8, 16]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh[h\\h=\x93\x94\x85\x94R\x94}\x94hbX\x12\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "15", \n        "span": "0", \n        "struct_info_": "10", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 8, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "6", \n        "span": "0", \n        "type_annotation": "7"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "11"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [12, 5, 13, 14]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "128"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe\x85\x94R\x94h3h#\x85\x94R\x94h3h\x02(h\x07(K\x02K\x00K\x00K\x04K\x06KSCDg\x00}\x02|\x00D\x00]\x19}\x03t\x00|\x03t\x01j\x02j\x03\x83\x02r\x14|\x02\xa0\x04|\x03j\x05\xa1\x01\x01\x00q\x04|\x02\xa0\x04d\x01|\x01d\x02\x16\x00\x13\x00\xa1\x01\x01\x00q\x04t\x06|\x02\x83\x01S\x00\x94NK\x02K\r\x87\x94(h8h9\x8c\x03tir\x94\x8c\x06IntImm\x94h>\x8c\x05value\x94\x8c\x05tuple\x94t\x94(h?h\n\x8c\x08produced\x94\x8c\x03var\x94t\x94h\x0fhJKKC\x0c\x04\x01\x08\x01\x0e\x01\x0e\x01\x14\x02\x08\x01\x94))t\x94R\x94h\x16NNNt\x94R\x94h&h{}\x94}\x94(h\x19hJh)\x8c)get_shape_gen_func.<locals>.produce_shape\x94h+}\x94(h?\x8c\x0etvm.relax.expr\x94\x8c\tShapeExpr\x94\x93\x94h\n\x8c\x08builtins\x94\x8c\x03int\x94\x93\x94\x8c\x06return\x94\x8c\t_operator\x94\x8c\x07getitem\x94\x93\x94\x8c\x06typing\x94\x8c\x05Tuple\x94\x93\x94h\x85\x86\x94R\x94uh-Nh.Nh/h\x1ah0Nh1N\x8c\x17_cloudpickle_submodules\x94]\x94h\x00\x8c\tsubimport\x94\x93\x94\x8c\x07tvm.tir\x94\x85\x94R\x94a\x8c\x0b__globals__\x94}\x94h9h\x92h9\x85\x94R\x94su\x86\x94\x86R0\x85\x94R\x94\x87\x94h\x8f]\x94(h\x92h\\\x85\x94R\x94h\x92\x8c\ttvm.relax\x94\x85\x94R\x94eh\x96}\x94h9h\x99su\x86\x94\x86R0\x85\x94R\x94h3h#\x85\x94R\x94\x86\x94h\x8f]\x94h\x96}\x94u\x86\x94\x86R0.')
# from tvm.script import tir as T

@T.prim_func
def main(var_A: T.handle, var_T_transpose: T.handle):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
            T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]
if __name__ == "__main__":
    bench = MLCBench()
    for _ in range(SAMPLE_NUMBER):
        input_infos, median, std = bench.benchmark(
            main, input_shape_gen_func, "llvm -num-cores=4", tvm.cpu()
        )
        bench.record(RELAX_FUNC_NAME, PRIM_FUNC_NAME, input_infos, median, std, WEIGHT, WEIGHT*median)
    bench.show()

