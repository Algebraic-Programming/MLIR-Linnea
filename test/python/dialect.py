# RUN: %PYTHON %s | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as linnea
from mlir_standalone.dialects import builtin as builtin
from mlir_standalone.dialects import func as func

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

# CHECK-LABEL: TEST: testAttributeEncoding
@run
def testAttributeEncoding():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    parsed = Attribute.parse(
      '#linnea.property<["general"]>')
    # CHECK: #linnea.property<["general"]>
    print(parsed)
    casted = linnea.MatrixEncodingAttr(parsed)
    # CHECK: equal: True
    print(f"equal: {casted == parsed}") 

# CHECK-LABEL: TEST: testAttributeEncodingOnTensor
@run
def testAttributeEncodingOnTensor():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    encoding = linnea.MatrixEncodingAttr(Attribute.parse(
      '#linnea.property<["general"]>'))
    tt = RankedTensorType.get((23,23), F32Type.get(), encoding=encoding)
    # CHECK: tensor<23x23xf32, #linnea.property
    print(tt)
    # CHECK: #linnea.property
    print(tt.encoding)
    assert(tt.encoding == encoding)

# CHECK-LABEL: TEST: testMatrixType
@run
def testMatrixType():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    parsed = Type.parse(
      '!linnea.matrix<#linnea.property<["general"]>, [32, 32], f32>')
    # CHECK: !linnea.matrix<#linnea.property<["general"]>, [32, 32], f32>
    print(parsed)
    casted = linnea.MatrixType(parsed)
    # CHECK: equal: True
    print(f"equal: {casted == parsed}")

# CHECK-LABEL: TEST: testParsing
@run
def testParsing():
  with Context() as ctx:
    linnea.register_dialect()
    module = Module.parse("""
      func @bar(%arg0: !linnea.matrix<#linnea.property<["general"]>,[32,32], f32>) {
        %0 = linnea.equation {
          %1 = linnea.transpose %arg0 : 
            !linnea.matrix<#linnea.property<["general"]>,[32,32], f32> -> !linnea.term
          linnea.yield %1 : !linnea.term
        }
        return 
      }
    """)
    # CHECK-LABEL: @bar(
    # CHECK: %{{.*}} = linnea.equation{
    # CHECK:    %[[T:.*]] = linnea.transpose %{{.*}}
    # CHECK:    linnea.yield %[[T]]
    # CHECK: }
    # CHECK: return
    print(str(module))

# CHECK-LABEL: TEST: buildLinneaTermType
@run
def buildLinneaTermType():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    tt = linnea.TermType.get(ctx)
    # CHECK: !linnea.term
    print(tt)

# CHECK-LABEL: TEST: buildMatrixAttribute
@run
def buildMatrixAttribute():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    p = [linnea.Property.general]
    attr = linnea.MatrixEncodingAttr.get(ctx, p)
    # CHECK: #linnea.property<["general"]>
    print(attr)

# CHECK-LABEL: TEST: buildMatrixType
@run
def buildMatrixType():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    p = [linnea.Property.general]
    attr = linnea.MatrixEncodingAttr.get(ctx, p)
    f32 = F32Type.get()
    matrix = linnea.MatrixType.get(ctx, attr, [23, 23], f32)
    # CHECK: !linnea.matrix<#linnea.property<["general"]>, [23, 23], f32> 
    print(matrix)

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
