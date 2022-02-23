# RUN: %PYTHON %s | FileCheck %s

from tools.frontend.utils import parse_input

TEXT = '''
n = 1500
m = 1000

Matrix X(n, m) <LowerTriangular>
Matrix Y(n, m) <>
Y = X * X
'''

equations = parse_input(TEXT)
# CHECK: Y = (X X)
for equation in equations:
  print(equations)
