
import tvm
import dill
from mlc_bench.benchmark import MLCBench
from tvm.script import tir as T

MODEL_NAME = "vicuna_v1_7b_fp_16"
FUNC_NAME = "softmax1"
FUNC_HASH = 7793420587337580373
WEIGHT = 32
CAT = -1
SAMPLE_NUMBER = 5

input_shape_gen_func = dill.loads(b'\x80\x04\x95]\x17\x00\x00\x00\x00\x00\x00\x8c\ndill._dill\x94\x8c\x10_create_function\x94\x93\x94(h\x00\x8c\x0c_create_code\x94\x93\x94(C\x04\x00\x01\x0e\x01\x94K\x00K\x00K\x00K\x02K\x04K\x1fC\x1c\x88\x01\x04\x00j\x00d\x017\x00\x02\x00_\x00\x88\x00|\x00i\x00|\x01\xa4\x01\x8e\x01S\x00\x94NK\x01\x86\x94\x8c\x05count\x94\x85\x94\x8c\x04args\x94\x8c\x06kwargs\x94\x86\x94\x8c$/home/ubuntu/mlc_bench/extraction.py\x94\x8c\x07wrapped\x94K#C\x04\x0e\x01\x0e\x01\x94\x8c\x04func\x94h\x0e\x86\x94)t\x94R\x94cextraction\n__dict__\nh\x0eNh\x00\x8c\x0c_create_cell\x94\x93\x94N\x85\x94R\x94h\x15N\x85\x94R\x94\x86\x94t\x94R\x94}\x94h\x08K\x00s}\x94(\x8c\x0f__annotations__\x94}\x94\x8c\x0c__qualname__\x94\x8c4get_shape_gen_func.<locals>.counted.<locals>.wrapped\x94u\x86\x94b\x8c\x08builtins\x94\x8c\x07getattr\x94\x93\x94\x8c\x04dill\x94\x8c\x05_dill\x94\x93\x94\x8c\x08_setattr\x94h$\x8c\x07setattr\x94\x93\x94\x87\x94R\x94h\x19\x8c\rcell_contents\x94h\x1c\x87\x94R0h.h\x17h/h\x02(h\x04(C.\x00\x02\x04\x01\x08\x01\x0e\x01\x02\x01\x10\x01\x04\x02\x0c\x01\x04\xfe\x02\xff\x06\x07\n\x01\x04\x02\x0c\x01\x04\xfe\x02\xff\x06\xff\x02\t\x08\x01\x0e\x01\x04\x01\x02\xff\x06\x06\x94K\x00K\x00K\x00K\x03K\x07K\x13C\x96g\x00}\x00\x88\x00D\x00]3}\x01t\x00|\x01t\x01j\x02j\x03\x83\x02r\x0eq\x04t\x00|\x01t\x01j\x02j\x04j\x05\x83\x02r$|\x00\xa0\x06\x88\x02|\x01j\x07\x88\x01j\x08\x83\x02|\x01j\tf\x02\xa1\x01\x01\x00q\x04|\x01j\nD\x00]\x0f}\x02|\x00\xa0\x06\x88\x02|\x02j\x07\x88\x01j\x08\x83\x02|\x02j\tf\x02\xa1\x01\x01\x00q\'q\x04\x88\x00D\x00]\x0e}\x01t\x00|\x01t\x01j\x02j\x03\x83\x02rH|\x00\xa0\x06d\x01\xa1\x01\x01\x00q:|\x00S\x00\x94N)\x8c\x05int64\x94\x86\x94\x86\x94(\x8c\nisinstance\x94\x8c\x03tvm\x94\x8c\x05relax\x94\x8c\x0fShapeStructInfo\x94\x8c\x0bstruct_info\x94\x8c\x10TensorStructInfo\x94\x8c\x06append\x94\x8c\x05shape\x94h\x08\x8c\x05dtype\x94\x8c\x06fields\x94t\x94\x8c\x07results\x94\x8c\x03arg\x94\x8c\x07sub_arg\x94\x87\x94h\r\x8c\x14input_shape_gen_func\x94K3C0\x04\x02\x08\x01\x0e\x01\x02\x01\x10\x01\x04\x01\x0c\x02\x04\x01\x02\xfe\x06\xff\n\x07\x04\x01\x0c\x02\x04\x01\x02\xfe\x06\xff\x02\xff\x08\t\x0e\x01\x04\x01\x02\x01\x04\xff\x02\x80\x04\x06\x94\x8c\tfunc_args\x94hE\x8c\rproduce_shape\x94\x87\x94)t\x94R\x94cextraction\n__dict__\nhENh\x15N\x85\x94R\x94h\x15N\x85\x94R\x94h\x15N\x85\x94R\x94\x87\x94t\x94R\x94}\x94}\x94(h\x1f}\x94h!\x8c0get_shape_gen_func.<locals>.input_shape_gen_func\x94u\x86\x94bh.hQh/h\x02(h\x04(C\x0c\x00\x01\x04\x01\x08\x01\x0e\x01\x0e\x02\x14\x01\x94K\x02K\x00K\x00K\x04K\x06KSCDg\x00}\x02|\x00D\x00]\x19}\x03t\x00|\x03t\x01j\x02j\x03\x83\x02r\x14|\x02\xa0\x04|\x03j\x05\xa1\x01\x01\x00q\x04|\x02\xa0\x04d\x01|\x01d\x02\x16\x00\x13\x00\xa1\x01\x01\x00q\x04t\x06|\x02\x83\x01S\x00\x94NK\x02K\x05\x87\x94(h6h7\x8c\x03tir\x94\x8c\x06IntImm\x94h<\x8c\x05value\x94\x8c\x05tuple\x94t\x94(h=h\x08\x8c\x08produced\x94\x8c\x03var\x94t\x94h\rhHK*C\x0c\x04\x01\x08\x01\x0e\x01\x0e\x01\x14\x02\x08\x01\x94))t\x94R\x94cextraction\n__dict__\nhHNNt\x94R\x94}\x94}\x94(h\x1f}\x94(h=\x8c\x0etvm.relax.expr\x94\x8c\tShapeExpr\x94\x93\x94h\x08h\x00\x8c\n_load_type\x94\x93\x94\x8c\x03int\x94\x85\x94R\x94\x8c\x06return\x94\x8c\t_operator\x94\x8c\x07getitem\x94\x93\x94\x8c\x06typing\x94\x8c\x05Tuple\x94\x93\x94ht\x86\x94R\x94uh!\x8c)get_shape_gen_func.<locals>.produce_shape\x94u\x86\x94b\x87\x94R0h.hOh/h\x1c\x87\x94R0h.hMh/]\x94(\x8c\x12tvm.runtime.object\x94\x8c\x0b_new_object\x94\x93\x94\x8c\x15tvm.relax.struct_info\x94\x8c\x0fTupleStructInfo\x94\x93\x94\x85\x94R\x94}\x94\x8c\x06handle\x94X\xcb\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TupleStructInfo", \n      "attrs": {\n        "fields": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [3]\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "4", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "17", \n        "span": "0", \n        "struct_info_": "12", \n        "values": "5"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [6, 7, 8, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "10", \n        "span": "0", \n        "type_annotation": "11"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "13"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [14, 15, 16, 9]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbh\x85h\x86h;\x93\x94\x85\x94R\x94}\x94h\x8cX\x0e\x08\x00\x00{\n  "root": 1, \n  "nodes": [\n    {\n      "type_key": ""\n    }, \n    {\n      "type_key": "relax.TensorStructInfo", \n      "attrs": {\n        "dtype": "float16", \n        "ndim": "4", \n        "shape": "2", \n        "span": "0"\n      }\n    }, \n    {\n      "type_key": "relax.expr.ShapeExpr", \n      "attrs": {\n        "_checked_type_": "15", \n        "span": "0", \n        "struct_info_": "10", \n        "values": "3"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [4, 5, 6, 7]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "tir.Var", \n      "attrs": {\n        "dtype": "int64", \n        "name": "8", \n        "span": "0", \n        "type_annotation": "9"\n      }\n    }, \n    {\n      "type_key": "runtime.String", \n      "repr_str": "n"\n    }, \n    {\n      "type_key": "PrimType", \n      "attrs": {"dtype": "int64"}\n    }, \n    {\n      "type_key": "relax.ShapeStructInfo", \n      "attrs": {\n        "ndim": "4", \n        "span": "0", \n        "values": "11"\n      }\n    }, \n    {\n      "type_key": "Array", \n      "data": [12, 13, 14, 7]\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "32"\n      }\n    }, \n    {\n      "type_key": "IntImm", \n      "attrs": {\n        "dtype": "int64", \n        "span": "0", \n        "value": "1"\n      }\n    }, \n    {\n      "type_key": "relax.ShapeType", \n      "attrs": {\n        "ndim": "4", \n        "span": "0"\n      }\n    }\n  ], \n  "b64ndarrays": [], \n  "attrs": {"tvm_version": "0.13.dev0"}\n}\x94sbe\x87\x94R0\x87\x94R0.')
# from tvm.script import tir as T

@T.prim_func
def main(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

if __name__ == "__main__":
    bench = MLCBench()
    for _ in range(SAMPLE_NUMBER):
        input_infos, median, std = bench.benchmark(
            main, input_shape_gen_func, "llvm -num-cores=4", tvm.cpu()
        )
        bench.record(FUNC_NAME, input_infos, median, std)
    bench.show()

