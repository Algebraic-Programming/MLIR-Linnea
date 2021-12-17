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

MatrixType getToggledType(MatrixType type,
                          MatrixType::MatrixProperty property) {
  ArrayRef<MatrixType::MatrixProperty> properties = type.getProperty();
  ArrayRef<MatrixType::MatrixProperty> newProperties = properties.drop_while(
      [&](MatrixType::MatrixProperty p) { return p == property; });
  assert(properties.size() == newProperties.size() + 1 && "property not found");

  SmallVector<MatrixType::MatrixProperty> newPropertiesVec = {
      newProperties.begin(), newProperties.end()};
  if (property == MatrixType::MatrixProperty::LowerTriangular)
    newPropertiesVec.push_back(MatrixType::MatrixProperty::UpperTriangular);
  else
    newPropertiesVec.push_back(MatrixType::MatrixProperty::LowerTriangular);

  SmallVector<int64_t, 2> dims = {type.getDims().begin(), type.getDims().end()};
  std::swap(dims[0], dims[1]);

  return MatrixType::get(type.getContext(), newPropertiesVec, dims,
                         type.getElementType());
}

void TransposeOp::build(OpBuilder &builder, OperationState &result,
                        Value input) {
  MatrixType inputType = input.getType().cast<MatrixType>();
  assert(inputType && "must be valid");

  Type outputType;
  if (isLowerTriangular(inputType)) {
    outputType =
        getToggledType(inputType, MatrixType::MatrixProperty::LowerTriangular);
  } else if (isUpperTriangular(inputType)) {
    outputType =
        getToggledType(inputType, MatrixType::MatrixProperty::UpperTriangular);
  } else {
    outputType = inputType;
  }
  build(builder, result, outputType, input);
}

void InverseOp::build(OpBuilder &builder, OperationState &result, Value input) {
  build(builder, result, input.getType(), input);
}

void CholeskyOp::build(OpBuilder &builder, OperationState &result,
                       Value input) {
  MatrixType inputType = input.getType().cast<MatrixType>();
  assert(inputType && "must be valid");
  assert(isSPD(inputType) && "expect SPD");
  ArrayRef<MatrixType::MatrixProperty> properties = inputType.getProperty();

  // preserve all properties but SPD.
  ArrayRef<MatrixType::MatrixProperty> newProperties =
      properties.drop_while([](MatrixType::MatrixProperty p) {
        return p == MatrixType::MatrixProperty::SPD;
      });

  // add new properties.
  SmallVector<MatrixType::MatrixProperty> newPropertiesVec = {
      newProperties.begin(), newProperties.end()};
  newPropertiesVec.push_back(MatrixType::MatrixProperty::UpperTriangular);
  newPropertiesVec.push_back(MatrixType::MatrixProperty::Factored);

  MatrixType newType =
      MatrixType::get(input.getContext(), newPropertiesVec, inputType.getDims(),
                      inputType.getElementType());

  build(builder, result, newType, input);
}

void MulOp::build(OpBuilder &builder, OperationState &result,
                  ValueRange inputs) {
  build(builder, result, Type(), inputs);
}

static LogicalResult verifyCholeskyOp(CholeskyOp op) { return success(); }

static LogicalResult verifyInverseOp(InverseOp op) { return success(); }

static LogicalResult verifyTransposeOp(TransposeOp op) { return success(); }

static LogicalResult verifyMulOp(MulOp op) { return success(); }

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

struct InverseOfMul : public OpRewritePattern<InverseOp> {
  using OpRewritePattern<InverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InverseOp op,
                                PatternRewriter &rewriter) const override {
    Value operand = op.getOperand();
    if (MulOp mul = operand.getDefiningOp<MulOp>()) {
      // expect a mul to have only two operands.
      if (mul.getOperands().size() != 2)
        return failure();
      SmallVector<Value, 4> resultInverse;
      for (auto operandMul : mul.getOperands()) {
        // only lower or upper triangular.
        if (!isLowerTriangular(operandMul.getType().cast<MatrixType>()) &&
            !isUpperTriangular(operandMul.getType().cast<MatrixType>()))
          return failure();
        resultInverse.push_back(
            rewriter.create<InverseOp>(op.getLoc(), operandMul)->getResult(0));
      }
      if (resultInverse.size() != 2)
        return failure();
      ArrayRef<int64_t> dimInverseOp1 =
          resultInverse[0].getType().cast<MatrixType>().getDims();
      ArrayRef<int64_t> dimInverseOp2 =
          resultInverse[1].getType().cast<MatrixType>().getDims();
      SmallVector<int64_t, 2> dimsMul = {dimInverseOp1[0], dimInverseOp2[1]};
      // Mul type is SPD.
      // TODO: worth moving these logic in the builder too?
      auto elementType =
          resultInverse[0].getType().cast<MatrixType>().getElementType();
      MatrixType newType =
          MatrixType::get(op.getContext(), MatrixType::MatrixProperty::SPD,
                          dimsMul, elementType);
      Value mulVal =
          rewriter.create<MulOp>(op.getLoc(), newType, resultInverse)
              ->getResult(0);
      rewriter.replaceOp(op, mulVal);
      return success();
    }
    return failure();
  }
};

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

struct CholeskyFact : public OpRewritePattern<InverseOp> {
  using OpRewritePattern<InverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InverseOp op,
                                PatternRewriter &rewriter) const override {
    Value operand = op.getOperand();
    Value result = op.getResult();
    // if operand is SPD and is used in a mul -> introduce cholesky fact.
    // 1. build matrix U
    // 2. build matrix trans(U)
    // 3. multiply the two : U(t) * U
    // 4. keep the inverse on their product

    // if the operand comes from another mul bail out.
    if (isa_and_nonnull<MulOp>(operand.getDefiningOp<MulOp>()))
      return failure();

    // check if we have SPD property and the users of the result
    // is a mul operation.
    if (isSPD(operand.getType().dyn_cast_or_null<MatrixType>()) &&
        llvm::any_of(result.getUsers(), [](Operation *user) {
          if (isa<MulOp>(user))
            return true;
          return false;
        })) {
      // U is upper trinagular.
      Value u = rewriter.create<CholeskyOp>(op.getLoc(), operand)->getResult(0);
      // U(T) is lower trinagular.
      Value uT = rewriter.create<TransposeOp>(op.getLoc(), u)->getResult(0);
      ArrayRef<int64_t> dimsUT = uT.getType().cast<MatrixType>().getDims();
      ArrayRef<int64_t> dimsU = u.getType().cast<MatrixType>().getDims();
      SmallVector<int64_t, 2> dimsMul = {dimsUT[0], dimsU[1]};
      auto elementType = uT.getType().cast<MatrixType>().getElementType();
      // Mul type is SPD.
      MatrixType newType =
          MatrixType::get(op.getContext(), MatrixType::MatrixProperty::SPD,
                          dimsMul, elementType);
      Value mul =
          rewriter.create<MulOp>(op.getLoc(), newType, ValueRange{uT, u})
              ->getResult(0);
      rewriter.updateRootInPlace(op, [&]() { op->setOperand(0, mul); });
      return success();
    }
    return failure();
  }
};

struct CollapseMul : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> candidates;
    for (auto operand : op.getOperands()) {
      if (MulOp parent = operand.getDefiningOp<MulOp>()) {
        Value parentResult = parent.getResult();
        if (!parentResult.hasOneUse())
          continue;
        Operation *childOp = *parentResult.getUsers().begin();
        if (childOp != op)
          continue;
        candidates.append(parent.getOperands().begin(),
                          parent.getOperands().end());

      } else {
        candidates.push_back(operand);
      }
    }
    if (candidates.size() == op.getOperands().size())
      return failure();
    rewriter.updateRootInPlace(op, [&]() { op->setOperands(candidates); });
    return success();
  }
};

} // end namespace

void InverseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<DoubleInverse, CholeskyFact, InverseOfMul>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<CollapseMul>(context);
}

void EquationOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {}
