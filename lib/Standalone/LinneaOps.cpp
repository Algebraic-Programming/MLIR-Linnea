//===- LinneaOps.cpp - Linnea dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linnea;

#define GET_OP_CLASSES
#include "Standalone/LinneaOps.cpp.inc"

LogicalResult InverseOp::canonicalize(InverseOp op, PatternRewriter &rewriter) {
  // Look the input chain of the current inverseOp.
  Value inverseInput = op.getOperand();
  InverseOp inverseInputOp = inverseInput.getDefiningOp<InverseOp>();

  // If the input comes from another transposeOp
  // simplify otherwise return.
  if (!inverseInputOp)
    return failure();

  rewriter.replaceOp(op, {inverseInputOp.getOperand()});
  return success();
}
