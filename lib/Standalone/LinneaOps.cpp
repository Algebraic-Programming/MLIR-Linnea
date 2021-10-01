//===- LinneaOps.cpp - Linnea dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaDialect.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linnea;

/// verify transpose properties (i.e., if input lower tri. output must be upper
/// tri.).
static LogicalResult verifySymbolicTransposeOp(SymbolicTransposeOp op) {
  Value input = op.getOperand();
  Value output = op.getResult();

  ArrayRef<MatrixType::MatrixProperty> propertiesInput =
      input.getType().cast<MatrixType>().getProperty();
  ArrayRef<MatrixType::MatrixProperty> propertiesOutput =
      output.getType().cast<MatrixType>().getProperty();

  if ((propertiesInput.size() == 1) &&
      (propertiesInput[0] == MatrixType::MatrixProperty::LowerTriangular) &&
      (propertiesOutput.size() != 1 ||
       propertiesOutput[0] != MatrixType::MatrixProperty::LowerTriangular)) {
    op.emitError(
        "input is lowerTriangular then output must be lowerTriangular");
    return failure();
  }

  return success();
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

struct CholeskyFact : public OpRewritePattern<InverseOp> {
  using OpRewritePattern<InverseOp>::OpRewritePattern;

  // Assume we are dealing with ranked 2d tensor type.
  Type getElementType(Value v) const {
    RankedTensorType t = v.getType().cast<RankedTensorType>();
    return t.getElementType();
  }

  // Assume we are dealing with ranked 2d tensor type.
  Value createBufferOpWithAttr(Value u, Value v, Location loc,
                               LinneaMatrixEncodingAttr attr,
                               PatternRewriter &rewriter) const {
    // static shape.
    mlir::Value buffer;
    RankedTensorType uT = u.getType().cast<RankedTensorType>();
    RankedTensorType vT = v.getType().cast<RankedTensorType>();
    if ((uT.hasStaticShape()) && (vT.hasStaticShape())) {
      auto shapeU = uT.getShape();
      auto shapeV = vT.getShape();
      SmallVector<int64_t, 2> outShape{shapeU[0], shapeV[1]};
      buffer = rewriter.create<linalg::InitTensorOp>(loc, outShape,
                                                     getElementType(u));
    }
    // dynamic shape.
    else {
      Value left = rewriter.createOrFold<tensor::DimOp>(loc, u, 0);
      Value right = rewriter.createOrFold<tensor::DimOp>(loc, v, 1);
      SmallVector<Value, 2> outShape{left, right};
      buffer = rewriter.create<linalg::InitTensorOp>(loc, outShape,
                                                     getElementType(u));
    }
    // cast
    RankedTensorType bufferType = buffer.getType().cast<RankedTensorType>();
    RankedTensorType castType = RankedTensorType::get(
        bufferType.getShape(), bufferType.getElementType(), attr);
    Value cast = rewriter.create<tensor::CastOp>(loc, castType, buffer);
    return cast;
  }

  LogicalResult matchAndRewrite(InverseOp op,
                                PatternRewriter &rewriter) const override {
    Value operand = op.getOperand();
    Value result = op.getResult();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    // if SPD and is used in a mul -> introduce cholesky fact.
    // 1. build matrix U
    // 2. build matrix trans(U)
    // 3. multiply the two : U(t) * U
    // 4. keep the inverse on their product

    // if the operand comes from another mul bail out.
    if (isa_and_nonnull<MulOp>(operand.getDefiningOp<MulOp>()))
      return failure();

    // check if we have SPD property and the users of the result
    // is a mul operation.
    if (isSPD(operand.getType()) &&
        llvm::any_of(result.getUsers(), [](Operation *user) {
          if (isa<MulOp>(user))
            return true;
          return false;
        })) {
      RankedTensorType opType = operand.getType().cast<RankedTensorType>();
      // cannot modify the attribute on the 'opType' thus recreate another type.
      RankedTensorType uType = RankedTensorType::get(
          opType.getShape(), opType.getElementType(),
          LinneaMatrixEncodingAttr::get(
              op->getContext(),
              {LinneaMatrixEncodingAttr::MatrixType::UpperTriangular,
               LinneaMatrixEncodingAttr::MatrixType::Factored}));
      // U is upper trinagular.
      Value u = rewriter.create<CholeskyOp>(op.getLoc(), uType, operand)
                    ->getResult(0);
      RankedTensorType uTType = RankedTensorType::get(
          opType.getShape(), opType.getElementType(),
          LinneaMatrixEncodingAttr::get(
              op->getContext(),
              {LinneaMatrixEncodingAttr::MatrixType::LowerTriangular,
               LinneaMatrixEncodingAttr::MatrixType::Factored}));
      // U(T) is lower trinagular.
      Value uT = rewriter.create<TransOp>(op.getLoc(), uTType, u)->getResult(0);
      // The mul between U(t) and U is SPD, thus adjust the buffer type.
      LinneaMatrixEncodingAttr mulAttr = LinneaMatrixEncodingAttr::get(
          op->getContext(), LinneaMatrixEncodingAttr::MatrixType::SPD);
      Value buffer =
          createBufferOpWithAttr(uT, u, op.getLoc(), mulAttr, rewriter);
      Value mul =
          rewriter
              .create<MulOp>(op.getLoc(), buffer.getType(), ValueRange{uT, u})
              ->getResult(0);

      rewriter.updateRootInPlace(op, [&]() { op->setOperand(0, mul); });
      return success();
    }
    return failure();
  }
};

} // end namespace

void InverseOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<DoubleInverse, CholeskyFact>(context);
}
