# RUN: %PYTHON %s | FileCheck %s

from tools.frontend.utils import parse_input

TEXT = '''
n = 1500
m = 1000

Matrix X(n, m) <FullRank>
ColumnVector y(n) <>
ColumnVector b(m) <>

b = inv(trans(X)*X)*trans(X)*y
'''
