# RUN: %PYTHON %s | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as sd

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

# CHECK-LABEL: TEST: testAttributeEncoding
@run
def testAttributeEncoding():
  with Context() as ctx, Location.unknown():
    sd.register_dialect()
    parsed = Attribute.parse(
      '#linnea.property<["general"]>')
    # CHECK: #linnea.property<["general"]>
    print(parsed)
    casted = sd.MatrixEncodingAttr(parsed)
    # CHECK: equal: True
    print(f"equal: {casted == parsed}") 

# CHECK-LABEL: TEST: testAttributeEncodingOnTensor
@run
def testAttributeEncodingOnTensor():
  with Context() as ctx, Location.unknown():
    sd.register_dialect()
    encoding = sd.MatrixEncodingAttr(Attribute.parse(
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
    sd.register_dialect()
    parsed = Type.parse(
      '!linnea.matrix<#linnea.property<["general"]>, [32, 32], f32>')
    # CHECK: !linnea.matrix<#linnea.property<["general"]>, [32, 32], f32>
    print(parsed)
    casted = sd.MatrixType(parsed)
    # CHECK: equal: True
    print(f"equal: {casted == parsed}")

# CHECK-LABEL: TEST: testParsing
@run
def testParsing():
  with Context() as ctx:
    sd.register_dialect()
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
