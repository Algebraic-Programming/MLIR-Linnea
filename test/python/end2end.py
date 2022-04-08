# RUN: SUPPORT_LIB=%llvmlibdir/libmlir_c_runner_utils%shlibext \
# RUN:    %PYTHON %s | FileCheck %s

import sys, os
from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as linnea
from mlir_standalone.dialects import builtin as builtin
from mlir_standalone.dialects import func as func
from mlir_standalone.execution_engine import *
from mlir_standalone.passmanager import *
from mlir_standalone.runtime import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

def lowerToLLVM(module): 
  import mlir_standalone.conversions 
  pm = PassManager.parse(
      "convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts")
  pm.run(module)
  return module

# CHECK-LABEL: TEST: testInvokeFloatAdd
@run
def testInvokeFloatAdd():
  with Context():
    module = Module.parse(r"""
func @add(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
  %add = arith.addf %arg0, %arg1 : f32
  return %add : f32
}
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    # Prepare arguments: two input floats and one result.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    arg0 = c_float_p(42.)
    arg1 = c_float_p(2.)
    res = c_float_p(-1.)
    execution_engine.invoke("add", arg0, arg1, res)
    # CHECK: 42.0 + 2.0 = 44.0
    print("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]))

# CHECK-LABEL: TEST: testInvokeVoid
@run
def testInvokeVoid():
  with Context():
    module = Module.parse(r"""
func @void() attributes { llvm.emit_c_interface } {
  return
}
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    # Nothing to check other than no exception thrown here.
    execution_engine.invoke("void")

# CHECK-LABEL: TEST: lowerMulOp
@run
def lowerMulOp():
  with Context() as ctx:
    linnea.register_dialect()
    module = Module.parse(r"""
func @entry() attributes { llvm.emit_c_interface } {
        
  %c5 = arith.constant 5 : index

  // Materialize and fill a linnea matrix.
  %fc = arith.constant 5.0 : f32
  %A = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
  %Af = linnea.fill(%fc, %A) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

  // Materialize and fill a linnea matrix.
  %B = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
  %Bf = linnea.fill(%fc, %B) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

  %0 = linnea.equation {
    %1 = linnea.mul.high %Af, %Bf { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> -> !linnea.term
    linnea.yield %1 : !linnea.term
  }
           
  linnea.print %0 : !linnea.term 
  return 
}""")

    support_lib = os.getenv('SUPPORT_LIB')
    assert support_lib is not None, 'SUPPORT_LIB must be defined'
    if not os.path.exists(support_lib):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

    pm = PassManager.parse("linnea-compiler")
    pm.run(module)
    execution_engine = ExecutionEngine(module, opt_level=0, shared_libs=[support_lib])
    # CHECK:      ( ( 25, 0, 0, 0, 0 ),
    # CHECK-SAME: ( 50, 25, 0, 0, 0 ),
    # CHECK-SAME: ( 75, 50, 25, 0, 0 ),
    # CHECK-SAME: ( 100, 75, 50, 25, 0 ),
    # CHECK-SAME: ( 125, 100, 75, 50, 25 ) )
    execution_engine.invoke("entry")
