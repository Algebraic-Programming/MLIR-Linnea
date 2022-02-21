#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._mlir_libs._standaloneDialects.standalone import fill_equation_region
from ..ir import *

class EquationOp:
  """Specialization for the equation op class."""

  def __init__(self, output = Type, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    results.append(output)
    op = self.build_generic(
      attributes = attributes,
      results = results,
      operands = operands,
      regions = regions,
      loc = loc,
      ip = ip)
    OpView.__init__(self, op)
    fill_equation_region(self.operation)

  def add_entry_block(self):
    return self.regions[0].blocks[0]
