# RUN: %PYTHON %s | FileCheck %s
import json
from tatsu.util import asjson
from tools import parser as p

TEXT = '''
Matrix L(n, n) <LowerTriangular, FullRank>

'''

ast = p.parseExpr(TEXT)
# CHECK: Matrix
print(json.dumps(asjson(ast), indent=2))
