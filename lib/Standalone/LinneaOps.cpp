//===- LinneaOps.cpp - Linnea dialect ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linnea;

static LogicalResult verifyInverseOp(InverseOp op) { return success(); }

static LogicalResult verifyTransposeOp(TransposeOp op) { return success(); }

static LogicalResult verifyMulOp(MulOpHigh op) { return success(); }

static LogicalResult verifyMulOp(MulOpLow op) { return success(); }

static ParseResult parseEquationOp(OpAsmParser &parser,
                                   OperationState &result) {
  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Parse terminator.
  Operation &yield = body->back().back();
  result.types.reserve(yield.getNumOperands());
  result.types.append(yield.operand_type_begin(), yield.operand_type_end());
  return success();
}

static void print(OpAsmPrinter &printer, EquationOp op) {
  // Print the region.
  printer.printRegion(op.getBody());
}

#define GET_OP_CLASSES
#include "Standalone/LinneaOps.cpp.inc"

namespace {

struct DoubleInverse : public OpRewritePattern<InverseOp> {
  using OpRewritePattern<InverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InverseOp op,
                                PatternRewriter &rewriter) const override {

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
};

} // end namespace

void InverseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<DoubleInverse>(context);
}
