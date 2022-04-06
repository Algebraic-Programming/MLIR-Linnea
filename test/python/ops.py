# RUN: %PYTHON %s | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as linnea
from mlir_standalone.dialects import arith
from mlir_standalone.dialects import func as func
from typing import List

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

# CHECK-LABEL: TEST: buildFuncOp
@run
def buildFuncOp():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    termType = linnea.TermType.get(ctx)
    p = [linnea.Property.general]
    f32 = F32Type.get()
    matrixType = linnea.MatrixType.get(ctx, linnea.MatrixEncodingAttr.get(ctx, p), [2, 2], f32)
    with InsertionPoint(module.body):
      f = func.FuncOp("some_func", ([termType, termType], []))
      with InsertionPoint(f.add_entry_block()):
        func.ReturnOp([])
      otherF = func.FuncOp("some_other_func", ([termType, matrixType], []))
      with InsertionPoint(otherF.add_entry_block()):
        func.ReturnOp([])
    
  # CHECK: module {
  # CHECK: func @some_func(%arg0: !linnea.term, %arg1: !linnea.term) {
  # CHECK:  return 
  # CHECK: }
  # CHECK: func @some_other_func(%arg0: !linnea.term, %arg1: !linnea.matrix<#linnea.property<["general"]>, [2, 2], f32>) {
  # CHECK: return 
  # CHECK: }
  # CHECK: }
  print(module)

# CHECK-LABEL: TEST: buildFillOp
@run
def buildFillOp():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    p = [linnea.Property.general]
    f32 = F32Type.get()
    matrixType = linnea.MatrixType.get(ctx, linnea.MatrixEncodingAttr.get(ctx, p), [2, 2], f32)
    with InsertionPoint(module.body):
      f = func.FuncOp("some_func", ([matrixType, f32], []))
      with InsertionPoint(f.add_entry_block()):
        linnea.FillOp(result = matrixType, value = f.arguments[1], output = f.arguments[0])
        func.ReturnOp([])
  # CHECK: module {
  # CHECK: func @some_func(%arg0: !linnea.matrix<#linnea.property<["general"]>, [2, 2], f32>, %arg1: f32) {
  # CHECK: %0 = linnea.fill(%arg1, %arg0) : f32, !linnea.matrix<#linnea.property<["general"]>, [2, 2], f32>
  # CHECK: return
  # CHECK: }
  # CHECK: }
  print(module)

# CHECK-LABEL: TEST: buildEquationOp
@run
def buildEquationOp():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    with InsertionPoint(module.body):
      termType = linnea.TermType.get(ctx)
      f = func.FuncOp("some_func", ([termType], []))
      with InsertionPoint(f.add_entry_block()):
        eqOp = linnea.EquationOp(termType)
        with InsertionPoint(eqOp.add_entry_block()):
          yieldOp = linnea.YieldOp(f.arguments[0])
        func.ReturnOp([])
  # CHECK: module {
  # CHECK: func @some_func(%arg0: !linnea.term) {
  # CHECK:   %0 = linnea.equation{
  # CHECK:     linnea.yield %arg0 : !linnea.term
  # CHECK:   }
  # CHECK: }
  # CHECK: }
  print(module)

# CHECK-LABEL: TEST: buildMulHighOp
@run
def buildMulHighOp():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    with InsertionPoint(module.body):
      termType = linnea.TermType.get(ctx)
      f = func.FuncOp("some_func", ([termType], []))
      with InsertionPoint(f.add_entry_block()):
        eqOp = linnea.EquationOp(termType)
        with InsertionPoint(eqOp.add_entry_block()):
          l = [f.arguments[0], f.arguments[0]]
          mulOp = linnea.MulOpHigh(termType, l)
          yieldOp = linnea.YieldOp(mulOp)
        func.ReturnOp([])
  # CHECK: {
  # CHECK: func @some_func(%arg0: !linnea.term) {
  # CHECK:  %0 = linnea.equation{
  # CHECK:    %1 = linnea.mul.high %arg0, %arg0 : !linnea.term, !linnea.term -> !linnea.term
  # CHECK:    linnea.yield %1 : !linnea.term  
  # CHECK:  }
  # CHECK: }
  # CHECK: }
  print(module)

# CHECK-LABEL: TEST: buildFullTest
@run
def buildFullTest():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    with InsertionPoint(module.body):
      f = func.FuncOp("entry", ([], []))
      with InsertionPoint(f.add_entry_block()):
        eqOp = linnea.EquationOp(linnea.TermType.get(ctx))
        with InsertionPoint(eqOp.add_entry_block()):
          f32 = F32Type.get()
          indexType = IndexType.get()
          fiveCstIdx = arith.ConstantOp(indexType, 5)
          fiveCst = arith.ConstantOp(f32, 5.0)  
          p = [linnea.Property.uppertriangular]
          matrixType = linnea.MatrixType.get(ctx, 
                                             linnea.MatrixEncodingAttr.get(ctx, p), [5, 5], f32) 
          aMatrix = linnea.AllocOp(matrixType, [fiveCstIdx, fiveCstIdx])
          aMatrixFilled = linnea.FillOp(matrixType, fiveCst, aMatrix)
          mulOp = linnea.MulOpHigh(linnea.TermType.get(ctx), [aMatrixFilled, aMatrixFilled])
          yieldOp = linnea.YieldOp(mulOp)
        func.ReturnOp([])
  # CHECK: {
  # CHECK: func @entry() {
  # CHECK:  %0 = linnea.equation{
  # CHECK:    %c5 = arith.constant 5 : index
  # CHECK:    %cst = arith.constant 5.000000e+00 : f32
  # CHECK:    %1 = linnea.alloc[%c5, %c5] : <#linnea.property<["upperTri"]>, [5, 5], f32>
  # CHECK:    %2 = linnea.fill(%cst, %1) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
  # CHECK:    %3 = linnea.mul.high %2, %2 : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32> -> !linnea.term
  # CHECK:    linnea.yield %3 : !linnea.term
  # CHECK:  }
  # CHECK: }
  # CHECK: }
  print(module)
