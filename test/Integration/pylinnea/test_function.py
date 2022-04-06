# RUN: %PYTHON %s | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as linnea
from mlir_standalone.dialects import builtin as builtin
from mlir_standalone.dialects import func as func

from tatsu.model import ModelBuilderSemantics
from tools.frontend.parser import LinneaParser
from tatsu.walkers import NodeWalker

TEXT = '''
n = 1500
m = 1000

Matrix X(n, m) <LowerTriangular>
Matrix Y(n, m) <>
Y = X * X
'''

def run(f):
  print("\nTEST: ", f.__name__)
  f()
  return f

class LinneaMLIRWalker(NodeWalker):
  def __init__(self, ctx, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._operand_types = dict()
    self._variables = dict()
    self._operands = dict()
    self._ctx = ctx

  @property
  def get_Operand_Types(self):
    return self._operand_types

  def walk_object(self, node):
    assert False, "This should never be reached unless the grammar is changed."

  def walk_Model(self, node):
    # FIXME: We parse two times avoid this.
    # First we parse operands and vars then we parse
    # equations. 
    if (len(self._operands) == 0):
      for var in node.vars:
        self.walk(var)
      for symbol in node.symbols:
        self.walk(symbol)
    else:
      for equation in node.equations:
        self.walk(equation)

  def walk_Size(self, node):
    self._variables[node.name] = int(node.value)

  def walk_Matrix(self, node):
    size = (self._variables[node.dims.rows], self._variables[node.dims.columns])
    p = []
    for prop in node.properties:
      if prop == 'LowerTriangular':
        p.append(linnea.Property.lowertriangular)
      if prop == 'UpperTriangular':
        p.append(linnea.Property.uppertriangular)
      if prop == 'General':
        p.append(linnea.Property.general)
    attr = linnea.MatrixEncodingAttr.get(self._ctx, p)
    # No type element information in Linnea. Use f32.
    f32 = F32Type.get()
    self._operand_types[node.name] = linnea.MatrixType.get(self._ctx, attr, size, f32)

  def walk_Symbol(self, node):
    return self._operands[node.name]

  def walk_Times(self, node):
    lhs = self.walk(node.left)
    rhs = self.walk(node.right)
    l = [lhs, rhs]
    termType = linnea.TermType.get(self._ctx)
    return linnea.MulOpHigh(termType, l)  

  # TODO: connect yielded value with lhs.
  def walk_Equation(self, node):
    termType = linnea.TermType.get(self._ctx)
    eqOp = linnea.EquationOp(termType)
    with InsertionPoint(eqOp.add_entry_block()):
      yielded = self.walk(node.rhs)
      yieldOp = linnea.YieldOp(yielded)

@run
def testBuildLinneaFromPythonArgs():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    parser = LinneaParser(semantics=ModelBuilderSemantics())
    ast = parser.parse(TEXT, rule_name = "model")
    walker = LinneaMLIRWalker(ctx)
    walker.walk(ast)
    operand_types = list(walker.get_Operand_Types.values())
    with InsertionPoint(module.body):
      f = func.FuncOp("some_func", (operand_types, []))
      with InsertionPoint(f.add_entry_block()):  
        func.ReturnOp([])
  # CHECK-LABEL: testBuildLinneaFromPythonArgs
  # CHECK: module {
  # CHECK: func @some_func(%arg0: !linnea.matrix<#linnea.property<["lowerTri"]>, [1500, 1000], f32>, %arg1: !linnea.matrix<#linnea.property<[]>, [1500, 1000], f32>) {
  # CHECK:  return
  # CHECK: }
  # CHECK: }
  print(module)

@run
def testBuildLinneaFromPythonBody(): 
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    parser = LinneaParser(semantics=ModelBuilderSemantics())
    ast = parser.parse(TEXT, rule_name = "model")
    walker = LinneaMLIRWalker(ctx)
    walker.walk(ast)
    operand_types = list(walker.get_Operand_Types.values())
    operand_ids = list(walker.get_Operand_Types.keys())
    with InsertionPoint(module.body):
      f = func.FuncOp("some_func", (operand_types, []))
      with InsertionPoint(f.add_entry_block()):
        assert(len(f.arguments) == len(operand_ids))
        zip_iterator = zip(operand_ids, f.arguments)
        walker._operands = dict(zip_iterator)
        walker.walk(ast)
        func.ReturnOp([])
    # CHECK-LABEL: testBuildLinneaFromPythonBody
    # CHECK: module {
    # CHECK: func @some_func(%arg0: !linnea.matrix<#linnea.property<["lowerTri"]>, [1500, 1000], f32>, %arg1: !linnea.matrix<#linnea.property<[]>, [1500, 1000], f32>) {
    # CHECK:  %0 = linnea.equation{
    # CHECK:  %1 = linnea.mul.high %arg0, %arg0 : !linnea.matrix<#linnea.property<["lowerTri"]>, [1500, 1000], f32>, !linnea.matrix<#linnea.property<["lowerTri"]>, [1500, 1000], f32> -> !linnea.term
    # CHECK:  linnea.yield %1 : !linnea.term
    # CHECK: }
    # CHECK: return
    # CHECK: }
    # CHECK: }

    print(module)
