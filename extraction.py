import tvm
import tvm.relax
from tvm.ir import IRModule
from mod_before_build import Module

prim_funcs = {}

for gv, func in Module.functions.items():
    if isinstance(func, tvm.tir.PrimFunc):
        prim_funcs[gv] = {"func": func, "args": []}

for gv, func in Module.functions.items():
    if isinstance(func, tvm.relax.Function):
        for block in func.body.blocks:
            for binding in block.bindings:
                if isinstance(binding.value, tvm.relax.expr.Call):
                    args = binding.value.args
                    gv = args[0]
                    if gv in prim_funcs:
                        assert isinstance(gv, tvm.ir.GlobalVar)
                        args = [arg.struct_info for arg in args[1:]] + [binding.value.struct_info]
                        new_args = True
                        for i in range(len(prim_funcs[gv]["args"])):
                            arg, count = prim_funcs[gv]["args"][i]
                            if arg == args:
                                prim_funcs[gv]["args"][i] = (arg, count + 1)
                                new_args = False
                                break
                        if new_args:
                            prim_funcs[gv]["args"].append((args, 1))


for gv in prim_funcs:
    print(gv, prim_funcs[gv])
