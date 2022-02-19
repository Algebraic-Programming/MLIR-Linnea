# RUN: %PYTHON %s | FileCheck %s

import json
from tatsu import parse
from tatsu.util import asjson

GRAMMAR = '''
    @@grammar::CALC


    start = expression $ ;


    expression
        =
        | expression '+' term
        | expression '-' term
        | term
        ;


    term
        =
        | term '*' factor
        | term '/' factor
        | factor
        ;


    factor
        =
        | '(' expression ')'
        | number
        ;


    number = /\d+/ ;
'''

ast = parse(GRAMMAR, '3 + 5')
# CHECK:      [
# CHECK-NEXT:   "3",
# CHECK-NEXT:   "+",
# CHECK-NEXT:   "5"
# CHECK-NEXT: ]
print(json.dumps(asjson(ast), indent=2))
