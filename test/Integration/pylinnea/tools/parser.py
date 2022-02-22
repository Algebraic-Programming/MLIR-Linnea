#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAVEAT UTILITOR
#
# This file was automatically generated by TatSu.
#
#    https://pypi.python.org/pypi/tatsu/
#
# Any changes you make to it will be overwritten the next time
# the file is generated.


from __future__ import print_function, division, absolute_import, unicode_literals

import sys

from tatsu.buffering import Buffer
from tatsu.parsing import Parser
from tatsu.parsing import tatsumasu, leftrec, nomemo
from tatsu.parsing import leftrec, nomemo  # noqa
from tatsu.util import re, generic_main  # noqa


KEYWORDS = {}  # type: ignore


class LinneaBuffer(Buffer):
    def __init__(
        self,
        text,
        whitespace=None,
        nameguard=None,
        comments_re=None,
        eol_comments_re='#.*?$',
        ignorecase=None,
        namechars='',
        **kwargs
    ):
        super(LinneaBuffer, self).__init__(
            text,
            whitespace=whitespace,
            nameguard=nameguard,
            comments_re=comments_re,
            eol_comments_re=eol_comments_re,
            ignorecase=ignorecase,
            namechars=namechars,
            **kwargs
        )


class LinneaParser(Parser):
    def __init__(
        self,
        whitespace=None,
        nameguard=None,
        comments_re=None,
        eol_comments_re='#.*?$',
        ignorecase=None,
        left_recursion=True,
        parseinfo=True,
        keywords=None,
        namechars='',
        buffer_class=LinneaBuffer,
        **kwargs
    ):
        if keywords is None:
            keywords = KEYWORDS
        super(LinneaParser, self).__init__(
            whitespace=whitespace,
            nameguard=nameguard,
            comments_re=comments_re,
            eol_comments_re=eol_comments_re,
            ignorecase=ignorecase,
            left_recursion=left_recursion,
            parseinfo=parseinfo,
            keywords=keywords,
            namechars=namechars,
            buffer_class=buffer_class,
            **kwargs
        )

    @tatsumasu()
    def _identifier_(self):  # noqa
        self._pattern('[a-zA-Z_][a-zA-Z0-9_]*')

    @tatsumasu()
    def _constant_(self):  # noqa
        self._pattern('[0-9]+(\\.[0-9]+)?([Ee][+-]?[0-9]+)?')

    @tatsumasu()
    def _integer_(self):  # noqa
        self._pattern('[0-9]+')

    @tatsumasu('Model')
    def _model_(self):  # noqa
        self._var_declarations_()
        self.name_last_node('vars')
        self._symbol_declarations_()
        self.name_last_node('symbols')
        self._equations_()
        self.name_last_node('equations')
        self._check_eof()
        self.ast._define(
            ['equations', 'symbols', 'vars'],
            []
        )

    @tatsumasu()
    def _var_declarations_(self):  # noqa

        def block0():
            self._var_declaration_()
            self.add_last_node_to_name('@')
        self._closure(block0)

    @tatsumasu()
    def _symbol_declarations_(self):  # noqa

        def block0():
            self._symbol_declaration_()
            self.add_last_node_to_name('@')
        self._closure(block0)

    @tatsumasu()
    def _equations_(self):  # noqa

        def block0():
            self._equation_()
            self.add_last_node_to_name('@')
        self._closure(block0)

    @tatsumasu('Size')
    def _var_declaration_(self):  # noqa
        self._identifier_()
        self.name_last_node('name')
        self._token('=')
        self._cut()
        self._integer_()
        self.name_last_node('value')
        self.ast._define(
            ['name', 'value'],
            []
        )

    @tatsumasu()
    def _symbol_declaration_(self):  # noqa
        with self._choice():
            with self._option():
                self._scalar_()
            with self._option():
                self._row_vector_()
            with self._option():
                self._column_vector_()
            with self._option():
                self._matrix_()
            with self._option():
                self._identity_matrix_()
            with self._option():
                self._zero_matrix_()
            self._error('no available options')

    @tatsumasu('Scalar')
    def _scalar_(self):  # noqa
        self._token('Scalar')
        self._cut()
        self._identifier_()
        self.name_last_node('name')
        self._token('<')

        def sep1():
            self._token(',')

        def block1():
            self._properties_()
            self.add_last_node_to_name('properties')
        self._join(block1, sep1)
        self._token('>')
        self.ast._define(
            ['name'],
            ['properties']
        )

    @tatsumasu('RowVector')
    def _row_vector_(self):  # noqa
        self._token('RowVector')
        self._cut()
        self._identifier_()
        self.name_last_node('name')
        self._dim_vector_()
        self.name_last_node('dims')
        self._token('<')

        def sep2():
            self._token(',')

        def block2():
            self._properties_()
            self.add_last_node_to_name('properties')
        self._join(block2, sep2)
        self._token('>')
        self.ast._define(
            ['dims', 'name'],
            ['properties']
        )

    @tatsumasu('ColumnVector')
    def _column_vector_(self):  # noqa
        self._token('ColumnVector')
        self._cut()
        self._identifier_()
        self.name_last_node('name')
        self._dim_vector_()
        self.name_last_node('dims')
        self._token('<')

        def sep2():
            self._token(',')

        def block2():
            self._properties_()
            self.add_last_node_to_name('properties')
        self._join(block2, sep2)
        self._token('>')
        self.ast._define(
            ['dims', 'name'],
            ['properties']
        )

    @tatsumasu('Matrix')
    def _matrix_(self):  # noqa
        self._token('Matrix')
        self._cut()
        self._identifier_()
        self.name_last_node('name')
        self._dim_matrix_()
        self.name_last_node('dims')
        self._token('<')

        def sep2():
            self._token(',')

        def block2():
            self._properties_()
            self.add_last_node_to_name('properties')
        self._join(block2, sep2)
        self._token('>')
        self.ast._define(
            ['dims', 'name'],
            ['properties']
        )

    @tatsumasu('IdentityMatrix')
    def _identity_matrix_(self):  # noqa
        self._token('IdentityMatrix')
        self._cut()
        self._identifier_()
        self.name_last_node('name')
        self._dim_matrix_()
        self.name_last_node('dims')
        self.ast._define(
            ['dims', 'name'],
            []
        )

    @tatsumasu('ZeroMatrix')
    def _zero_matrix_(self):  # noqa
        self._token('ZeroMatrix')
        self._cut()
        self._identifier_()
        self.name_last_node('name')
        self._dim_matrix_()
        self.name_last_node('dims')
        self.ast._define(
            ['dims', 'name'],
            []
        )

    @tatsumasu()
    def _dim_vector_(self):  # noqa
        self._token('(')
        self._identifier_()
        self.name_last_node('length')
        self._token(')')
        self.ast._define(
            ['length'],
            []
        )

    @tatsumasu()
    def _dim_matrix_(self):  # noqa
        self._token('(')
        self._identifier_()
        self.name_last_node('rows')
        self._token(',')
        self._identifier_()
        self.name_last_node('columns')
        self._token(')')
        self.ast._define(
            ['columns', 'rows'],
            []
        )

    @tatsumasu()
    def _properties_(self):  # noqa
        with self._choice():
            with self._option():
                self._token('Square')
                self.name_last_node('@')
            with self._option():
                self._token('SPD')
                self.name_last_node('@')
            with self._option():
                self._token('ColumnPanel')
                self.name_last_node('@')
            with self._option():
                self._token('RowPanel')
                self.name_last_node('@')
            with self._option():
                self._token('Diagonal')
                self.name_last_node('@')
            with self._option():
                self._token('Tridiagonal')
                self.name_last_node('@')
            with self._option():
                self._token('Banded')
                self.name_last_node('@')
            with self._option():
                self._token('LowerTriangular')
                self.name_last_node('@')
            with self._option():
                self._token('UpperTriangular')
                self.name_last_node('@')
            with self._option():
                self._token('UnitDiagonal')
                self.name_last_node('@')
            with self._option():
                self._token('Symmetric')
                self.name_last_node('@')
            with self._option():
                self._token('Hessenberg')
                self.name_last_node('@')
            with self._option():
                self._token('Orthogonal')
                self.name_last_node('@')
            with self._option():
                self._token('FullRank')
                self.name_last_node('@')
            with self._option():
                self._token('Non-singular')
                self.name_last_node('@')
            with self._option():
                self._token('Positive')
                self.name_last_node('@')
            with self._option():
                self._token('SPSD')
                self.name_last_node('@')
            with self._option():
                self._token('OrthogonalColumns')
                self.name_last_node('@')
            with self._option():
                self._token('OrthogonalRows')
                self.name_last_node('@')
            with self._option():
                self._token('Permutation')
                self.name_last_node('@')
            self._error('no available options')

    @tatsumasu('Equation')
    def _equation_(self):  # noqa
        self._symbol_()
        self.name_last_node('lhs')
        self._token('=')
        self._cut()
        self._expression_()
        self.name_last_node('rhs')
        self.ast._define(
            ['lhs', 'rhs'],
            []
        )

    @tatsumasu()
    @nomemo
    def _expression_(self):  # noqa
        with self._choice():
            with self._option():
                self._addition_()
            with self._option():
                self._subtraction_()
            with self._option():
                self._term_()
            self._error('no available options')

    @tatsumasu('Plus')
    @nomemo
    def _addition_(self):  # noqa
        self._term_()
        self.name_last_node('left')
        self._token('+')
        self.name_last_node('op')
        self._cut()
        self._expression_()
        self.name_last_node('right')
        self.ast._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu('Subtract')
    def _subtraction_(self):  # noqa
        self._term_()
        self.name_last_node('left')
        self._token('-')
        self.name_last_node('op')
        self._cut()
        self._expression_()
        self.name_last_node('right')
        self.ast._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu()
    @nomemo
    def _term_(self):  # noqa
        with self._choice():
            with self._option():
                self._multiplication_()
            with self._option():
                self._factor_()
            self._error('no available options')

    @tatsumasu('Times')
    @nomemo
    def _multiplication_(self):  # noqa
        self._factor_()
        self.name_last_node('left')
        self._token('*')
        self.name_last_node('op')
        self._cut()
        self._term_()
        self.name_last_node('right')
        self.ast._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu()
    @leftrec
    def _factor_(self):  # noqa
        with self._choice():
            with self._option():
                self._subexpression_()
            with self._option():
                self._transposed_expr_()
            with self._option():
                self._inverted_expr_()
            with self._option():
                self._minus_expr_()
            with self._option():
                self._literal_expr_()
            with self._option():
                self._symbol_()
            self._error('no available options')

    @tatsumasu()
    def _subexpression_(self):  # noqa
        self._token('(')
        self._cut()
        self._expression_()
        self.name_last_node('@')
        self._token(')')

    @tatsumasu('Transpose')
    @nomemo
    def _transposed_expr_(self):  # noqa
        with self._choice():
            with self._option():
                self._token('trans(')
                self._expression_()
                self.name_last_node('arg')
                self._token(')')
            with self._option():
                self._factor_()
                self.name_last_node('arg')
                self._token("'")
            self._error('no available options')
        self.ast._define(
            ['arg'],
            []
        )

    @tatsumasu('Inverse')
    def _inverted_expr_(self):  # noqa
        self._token('inv(')
        self._expression_()
        self.name_last_node('arg')
        self._token(')')
        self.ast._define(
            ['arg'],
            []
        )

    @tatsumasu('Minus')
    def _minus_expr_(self):  # noqa
        self._token('-')
        self._factor_()
        self.name_last_node('arg')
        self.ast._define(
            ['arg'],
            []
        )

    @tatsumasu('Number')
    def _literal_expr_(self):  # noqa
        self._constant_()
        self.name_last_node('value')
        self.ast._define(
            ['value'],
            []
        )

    @tatsumasu('Symbol')
    def _symbol_(self):  # noqa
        self._identifier_()
        self.name_last_node('name')
        self.ast._define(
            ['name'],
            []
        )


class LinneaSemantics(object):
    def identifier(self, ast):  # noqa
        return ast

    def constant(self, ast):  # noqa
        return ast

    def integer(self, ast):  # noqa
        return ast

    def model(self, ast):  # noqa
        return ast

    def var_declarations(self, ast):  # noqa
        return ast

    def symbol_declarations(self, ast):  # noqa
        return ast

    def equations(self, ast):  # noqa
        return ast

    def var_declaration(self, ast):  # noqa
        return ast

    def symbol_declaration(self, ast):  # noqa
        return ast

    def scalar(self, ast):  # noqa
        return ast

    def row_vector(self, ast):  # noqa
        return ast

    def column_vector(self, ast):  # noqa
        return ast

    def matrix(self, ast):  # noqa
        return ast

    def identity_matrix(self, ast):  # noqa
        return ast

    def zero_matrix(self, ast):  # noqa
        return ast

    def dim_vector(self, ast):  # noqa
        return ast

    def dim_matrix(self, ast):  # noqa
        return ast

    def properties(self, ast):  # noqa
        return ast

    def equation(self, ast):  # noqa
        return ast

    def expression(self, ast):  # noqa
        return ast

    def addition(self, ast):  # noqa
        return ast

    def subtraction(self, ast):  # noqa
        return ast

    def term(self, ast):  # noqa
        return ast

    def multiplication(self, ast):  # noqa
        return ast

    def factor(self, ast):  # noqa
        return ast

    def subexpression(self, ast):  # noqa
        return ast

    def transposed_expr(self, ast):  # noqa
        return ast

    def inverted_expr(self, ast):  # noqa
        return ast

    def minus_expr(self, ast):  # noqa
        return ast

    def literal_expr(self, ast):  # noqa
        return ast

    def symbol(self, ast):  # noqa
        return ast

def parseExpr(text):
  parser = LinneaParser()
  ast = parser.parse(text, rule_name='identifier')
  return ast 