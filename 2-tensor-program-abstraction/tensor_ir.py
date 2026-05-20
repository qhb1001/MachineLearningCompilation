import numpy as np
import tvm  
from tvm.ir.module import IRModule
from tvm.script import tirx as T

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm_relu = np.maximum(a_np @ b_np, 0)

def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

import IPython

# print(IPython.display.Code(MyModule.script(), language="python"))

a_nd = tvm.runtime.tensor(a_np)
b_nd = tvm.runtime.tensor(b_np)
c_nd = tvm.runtime.tensor(np.empty((128, 128), dtype="float32"))

def sample_code(): 
    sch = tvm.s_tir.Schedule(MyModuleWithAxisRemapSugar)
    block_Y = sch.get_sblock("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, 4])
    sch.reorder(j0, k, j1)
    block_C = sch.get_sblock("C", func_name="mm_relu")
    sch.reverse_compute_at(block_C, j0)
    sch.decompose_reduction(block_Y, k)
    script_code = IPython.display.Code(sch.mod.script(), language="python")
    print(script_code)

    rt_lib = tvm.compile(MyModule, target="llvm")
    type(c_nd)
    func_mm_relu = rt_lib["mm_relu"]
    func_mm_relu(a_nd, b_nd, c_nd)

    np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

    rt_lib_after = tvm.compile(sch.mod, target="llvm")
    rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

    f_timer_before = rt_lib.mod.time_evaluator("mm_relu", tvm.cpu())
    print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
    f_timer_after = rt_lib_after.mod.time_evaluator("mm_relu", tvm.cpu())
    print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)


def transform(mod, jfactor):
    sch = tvm.s_tir.Schedule(mod)
    block_Y = sch.get_sblock("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_sblock("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod

def evaludate_transoformation(jfactor: Int):
    mod_transformed = transform(MyModule, jfactor)

    rt_lib_transformed = tvm.compile(mod_transformed, "llvm")
    f_timer_transformed = rt_lib_transformed.mod.time_evaluator("mm_relu", tvm.cpu())
    mean_time = f_timer_transformed(a_nd, b_nd, c_nd).mean
    print(f"Time cost of transformed mod_transformed {mean_time} sec for jfactor of {jfactor}")
    return mean_time

minimum_duration = None
final_j = None
for j in range(1, 128):
    experiment_result = evaludate_transoformation(j)
    if minimum_duration is None:
        minimum_duration = experiment_result
        final_j = j
    elif minimum_duration > experiment_result:
        minimum_duration = experiment_result
        final_j = j

print(f"The final j is {final_j}")