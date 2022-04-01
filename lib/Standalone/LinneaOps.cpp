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

//===----------------------------------------------------------------------===//
// CastToBuiltinTensor
//===----------------------------------------------------------------------===//

OpFoldResult CastToBuiltinTensorOp::fold(ArrayRef<Attribute> attr) {
  // cast_to_builtin_tensor(cast_from_builtin_tensor(a)) -> a
  if (auto castFromBuiltinTensor =
          input().getDefiningOp<CastFromBuiltinTensorOp>())
    if (castFromBuiltinTensor.output().getType() == input().getType())
      return castFromBuiltinTensor.input();
  return {};
}

namespace {
struct NoopCastToBuiltin : public OpRewritePattern<CastToBuiltinTensorOp> {
  using OpRewritePattern<CastToBuiltinTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastToBuiltinTensorOp op,
                                PatternRewriter &rewriter) const override {
    Type inputType = op.input().getType();
    Type outputType = op.output().getType();
    if (inputType == outputType) {
      op.output().replaceAllUsesWith(op.input());
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
} // namespace

void CastToBuiltinTensorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.insert<NoopCastToBuiltin>(context);
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//
LogicalResult FillOp::verify() { return success(); }

void FillOp::build(OpBuilder &builder, OperationState &result, Value value,
                   Value output) {
  build(builder, result, output.getType(), value, output);
}

void FillOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer << value() << ", ";
  printer << output();
  printer << ") : ";
  printer << value().getType();
  printer << ", ";
  printer << output().getType();
}

ParseResult FillOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseLParen())
    return failure();

  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return failure();

  if (parser.parseComma())
    return failure();

  OpAsmParser::UnresolvedOperand output;
  if (parser.parseOperand(output))
    return failure();

  if (parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseColon())
    return failure();

  mlir::Type typeOperand;
  if (parser.parseCustomTypeWithFallback(typeOperand))
    return failure();

  if (parser.parseComma())
    return failure();

  mlir::Type typeOutput;
  if (parser.parseCustomTypeWithFallback(typeOutput))
    return failure();

  // cannot use assembly format as I need to add the result type
  // to generate a linalg.fillOp.
  result.addTypes(typeOutput);

  if (parser.resolveOperand(operand, typeOperand, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperand(output, typeOutput, result.operands))
    return ::mlir::failure();

  return success();
}

//===----------------------------------------------------------------------===//
// InverseOpHigh
//===----------------------------------------------------------------------===//
LogicalResult InverseOpHigh::verify() { return success(); }

//===----------------------------------------------------------------------===//
// InverseOpLow
//===----------------------------------------------------------------------===//
LogicalResult InverseOpLow::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
LogicalResult TransposeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// MulOp (low/high)
//===----------------------------------------------------------------------===//
LogicalResult MulOpHigh::verify() { return success(); }
LogicalResult MulOpLow::verify() { return success(); }

//===----------------------------------------------------------------------===//
// AddOp (low/high)
//===----------------------------------------------------------------------===//
LogicalResult AddOpHigh::verify() { return success(); }
LogicalResult AddOpLow::verify() { return success(); }

//===----------------------------------------------------------------------===//
// EquationOp
//===----------------------------------------------------------------------===//
ParseResult EquationOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  if (body->empty()) {
    parser.emitError(parser.getNameLoc(), "Unexpected empty region");
    return failure();
  }

  // Parse terminator.
  Operation &yield = body->back().back();
  result.types.reserve(yield.getNumOperands());
  result.types.append(yield.operand_type_begin(), yield.operand_type_end());
  return success();
}

void EquationOp::print(OpAsmPrinter &printer) {
  // Print the region.
  printer.printRegion(getRegion());
}

#define GET_OP_CLASSES
#include "Standalone/LinneaOps.cpp.inc"

namespace {

struct DoubleInverse : public OpRewritePattern<InverseOpLow> {
  using OpRewritePattern<InverseOpLow>::OpRewritePattern;

  LogicalResult matchAndRewrite(InverseOpLow op,
                                PatternRewriter &rewriter) const override {

    // Look the input chain of the current inverseOp.
    Value inverseInput = op.getOperand();
    InverseOpLow inverseInputOp = inverseInput.getDefiningOp<InverseOpLow>();

    // If the input comes from another transposeOp
    // simplify otherwise return.
    if (!inverseInputOp)
      return failure();

    rewriter.replaceOp(op, {inverseInputOp.getOperand()});
    return success();
  }
};

} // end namespace

void InverseOpLow::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<DoubleInverse>(context);
}

namespace {

struct MultiplyWithIdentity : public OpRewritePattern<MulOpLow> {
  using OpRewritePattern<MulOpLow>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOpLow op,
                                PatternRewriter &rewriter) const override {
    Type left = op.left().getType();
    Type right = op.right().getType();
    if (auto it = left.dyn_cast_or_null<IdentityType>()) {
      Value result = op.output();
      result.replaceAllUsesWith(op.right());
      rewriter.eraseOp(op);
      return success();
    }
    if (auto it = right.dyn_cast_or_null<IdentityType>()) {
      Value result = op.output();
      result.replaceAllUsesWith(op.left());
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // end namespace

void MulOpLow::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<MultiplyWithIdentity>(context);
}
