# RUN: %PYTHON %s | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone as linnea
from mlir_standalone.dialects import builtin as builtin
from mlir_standalone.dialects import std as std

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
    self._symbols = []
    self._variables = dict()
    self._ctx = ctx;

  @property
  def symbols(self):
    return self._symbols

  def walk_object(self, node):
    assert False, "This should never be reached unless the grammar is changed."

  def walk_Model(self, node):
    for var in node.vars:
      self.walk(var)
    for symbol in node.symbols:
      self.walk(symbol)

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
    self._symbols.append(linnea.MatrixType.get(self._ctx, attr, size, f32))

@run
def testBuildLinneaFromPython():
  with Context() as ctx, Location.unknown():
    linnea.register_dialect()
    module = Module.create()
    parser = LinneaParser(semantics=ModelBuilderSemantics())
    ast = parser.parse(TEXT, rule_name = "model")
    walker = LinneaMLIRWalker(ctx)
    walker.walk(ast)
    operands = walker.symbols
    with InsertionPoint(module.body):
      func = builtin.FuncOp("some_func", (operands, []))
      with InsertionPoint(func.add_entry_block()):  
        std.ReturnOp([])
  # CHECK: module {
  # CHECK: func @some_func(%arg0: !linnea.matrix<#linnea.property<["lowerTri"]>, [1500, 1000], f32>, %arg1: !linnea.matrix<#linnea.property<[]>, [1500, 1000], f32>) {
  # CHECK:  return
  # CHECK: }
  # CHECK: }

  print(module)  
