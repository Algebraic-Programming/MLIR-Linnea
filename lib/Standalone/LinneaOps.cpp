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
  Value createBufferOpFor(Value u, Value v, Location loc,
                          PatternRewriter &rewriter) const {
    // static shape.
    RankedTensorType uT = u.getType().cast<RankedTensorType>();
    RankedTensorType vT = v.getType().cast<RankedTensorType>();
    if ((uT.hasStaticShape()) && (vT.hasStaticShape())) {
      auto shapeU = uT.getShape();
      auto shapeV = vT.getShape();
      SmallVector<int64_t, 2> outShape{shapeU[0], shapeV[1]};
      Value buffer = rewriter.create<linalg::InitTensorOp>(loc, outShape,
                                                           getElementType(u));
      return buffer;
    }
    // dynamic shape.
    Value left = rewriter.createOrFold<tensor::DimOp>(loc, u, 0);
    Value right = rewriter.createOrFold<tensor::DimOp>(loc, v, 1);
    SmallVector<Value, 2> outShape{left, right};
    Value buffer =
        rewriter.create<linalg::InitTensorOp>(loc, outShape, getElementType(u));
    return buffer;
  }

  LogicalResult matchAndRewrite(InverseOp op,
                                PatternRewriter &rewriter) const override {
    Value operand = op.getOperand();
    Value result = op.getResult();

    // if SPD and is used in a mul -> introduce cholesky fact.
    // 1. build matrix u
    // 2. build matrix trans(u)
    // 3. multiply the two.
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
              LinneaMatrixEncodingAttr::MatrixType::UpperTriangular));
      Value u = rewriter.create<CholeskyOp>(op.getLoc(), uType, operand)
                    ->getResult(0);
      RankedTensorType uTType = RankedTensorType::get(
          opType.getShape(), opType.getElementType(),
          LinneaMatrixEncodingAttr::get(
              op->getContext(),
              LinneaMatrixEncodingAttr::MatrixType::LowerTriangular));
      Value uT = rewriter.create<TransOp>(op.getLoc(), uTType, u)->getResult(0);
      // create a static buffer if possible.
      Value buffer = createBufferOpFor(u, uT, op.getLoc(), rewriter);
      Value mul =
          rewriter
              .create<MulOp>(op.getLoc(), buffer.getType(), ValueRange{u, uT})
              ->getResult(0);
      Value iMul = rewriter.create<InverseOp>(op.getLoc(), mul.getType(), mul)
                       ->getResult(0);
      rewriter.replaceOp(op, iMul);
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
