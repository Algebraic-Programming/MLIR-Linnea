# RUN: %PYTHON %s | FileCheck %s

import sys
from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as linnea
from mlir_standalone.dialects import builtin as builtin
from mlir_standalone.dialects import std as std
from mlir_standalone.execution_engine import *
from mlir_standalone.passmanager import *
from mlir_standalone.runtime import *

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

def lowerToLLVM(module): 
  import mlir_standalone.conversions 
  pm = PassManager.parse(
      "convert-memref-to-llvm,convert-std-to-llvm,reconcile-unrealized-casts")
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
    log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]))

# CHECK-LABEL: TEST: testInvokedVoid
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
