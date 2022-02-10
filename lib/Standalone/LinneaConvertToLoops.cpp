//===- LinneaConvertToLoops.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaPasses.h"
#include "Standalone/LinneaUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linnea;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-passes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

static void buildLoopNestImpl(PatternRewriter &rewriter, linalg::FillOp op,
                              Value destMemRef) {
  RankedTensorType outputTensor =
      op.output().getType().cast<RankedTensorType>();
  Location loc = op->getLoc();
  SmallVector<Value> ubs;
  Value dim1 =
      rewriter.create<arith::ConstantIndexOp>(loc, outputTensor.getShape()[0]);
  Value dim2 =
      rewriter.create<arith::ConstantIndexOp>(loc, outputTensor.getShape()[1]);
  ubs = {dim1, dim2};
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lbs = {zero, zero};
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> steps = {one, one};

  (void)scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps,
      [&](OpBuilder &b, Location loc, ValueRange localIvs) {
        b.create<memref::StoreOp>(loc, op.value(), destMemRef, localIvs);
      });
}

static Value constantZero(OpBuilder &builder, Location loc, Type tp) {
  return builder.create<arith::ConstantOp>(loc, tp, builder.getZeroAttr(tp));
}

static void buildLoopNestTriangularImpl(PatternRewriter &rewriter,
                                        linalg::FillOp op, Value destMemRef,
                                        arith::CmpIPredicate predicate) {
  RankedTensorType outputTensor =
      op.output().getType().cast<RankedTensorType>();
  Location loc = op->getLoc();
  SmallVector<Value> ubs;
  Value dim1 =
      rewriter.create<arith::ConstantIndexOp>(loc, outputTensor.getShape()[0]);
  Value dim2 =
      rewriter.create<arith::ConstantIndexOp>(loc, outputTensor.getShape()[1]);
  ubs = {dim1, dim2};
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lbs = {zero, zero};
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> steps = {one, one};

  Value zeroValue = constantZero(
      rewriter, loc, destMemRef.getType().cast<MemRefType>().getElementType());
  // We still assume a rectangular iteration domain, thus set to zero
  // element in the upper-part.
  (void)scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps,
      [&](OpBuilder &b, Location loc, ValueRange localIvs) {
        Value outerLoopIvCast =
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), localIvs[0]);
        Value innerLoopIvCast =
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), localIvs[1]);
        Value cond = b.create<arith::CmpIOp>(loc, predicate, outerLoopIvCast,
                                             innerLoopIvCast);
        auto ifOp = b.create<scf::IfOp>(loc, cond,
                                        /*hasElseRegion*/ true);
        b.setInsertionPointToStart(&ifOp.getThenRegion().front());
        b.create<memref::StoreOp>(loc, op.value(), destMemRef, localIvs);
        b.setInsertionPointToStart(&ifOp.getElseRegion().front());
        b.create<memref::StoreOp>(loc, zeroValue, destMemRef, localIvs);
        b.setInsertionPointAfter(ifOp);
      });
}

static void buildLoopNest(PatternRewriter &rewriter, linalg::FillOp op,
                          Value destMemRef) {
  ArrayRef<LinneaMatrixEncodingAttr::MatrixProperty> encoding =
      getLinneaTensorEncoding(op.output().getType()).getEncoding();
  if (encoding.size() == 1 &&
      encoding[0] ==
          LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular) {
    buildLoopNestTriangularImpl(rewriter, op, destMemRef,
                                arith::CmpIPredicate::sge);
    return;
  }
  if (encoding.size() == 1 &&
      encoding[0] ==
          LinneaMatrixEncodingAttr::MatrixProperty::UpperTriangular) {
    buildLoopNestTriangularImpl(rewriter, op, destMemRef,
                                arith::CmpIPredicate::sle);
    return;
  }
  buildLoopNestImpl(rewriter, op, destMemRef);
}

static Value castToMemRef(Value val, PatternRewriter &rewriter, Location loc) {
  RankedTensorType tensor = val.getType().cast<RankedTensorType>();
  MemRefType memref =
      MemRefType::get(tensor.getShape(), tensor.getElementType());
  Type castedLinneaTensorType =
      RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  Value castedLinneaTensor =
      rewriter.create<CastToBuiltinTensorOp>(loc, castedLinneaTensorType, val);
  Value dest = rewriter.create<bufferization::ToMemrefOp>(loc, memref,
                                                          castedLinneaTensor);
  return dest;
}

static Value castToTensor(Value val, PatternRewriter &rewriter, Location loc) {
  RankedTensorType tensor = val.getType().cast<RankedTensorType>();
  Type castedLinneaTensorType =
      RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  Value castedLinneaTensor =
      rewriter.create<CastToBuiltinTensorOp>(loc, castedLinneaTensorType, val);
  return castedLinneaTensor;
}

struct InitOpConverter : public OpRewritePattern<linalg::InitTensorOp> {
public:
  using OpRewritePattern<linalg::InitTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::InitTensorOp op,
                                PatternRewriter &rewriter) const override {
    Type output = op.result().getType();
    RankedTensorType outputTensor = output.cast<RankedTensorType>();
    auto encoding = getLinneaTensorEncoding(output);
    if (!encoding)
      return failure();

    Location loc = op->getLoc();
    MemRefType memTp =
        MemRefType::get(outputTensor.getShape(), outputTensor.getElementType());
    Value memref = rewriter.create<memref::AllocaOp>(loc, memTp);
    RankedTensorType builtinTensor = RankedTensorType::get(
        outputTensor.getShape(), outputTensor.getElementType());
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, builtinTensor,
                                                           memref);
    return success();
  }
};

struct FillOpConverter : public OpRewritePattern<linalg::FillOp> {
public:
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: should we fail if fillop cannot be bufferize in place?
    Type output = op.output().getType();
    RankedTensorType outputTensor = output.cast<RankedTensorType>();
    auto encoding = getLinneaTensorEncoding(output);
    if (!encoding)
      return failure();

    Location loc = op->getLoc();
    Value dest = castToMemRef(op.output(), rewriter, loc);
    buildLoopNest(rewriter, op, dest);

    RankedTensorType builtinTensorType = RankedTensorType::get(
        outputTensor.getShape(), outputTensor.getElementType());
    Value ret = rewriter.create<bufferization::ToTensorOp>(
        loc, builtinTensorType, dest);
    rewriter.replaceOpWithNewOp<CastFromBuiltinTensorOp>(
        op, op.output().getType(), ret);
    return success();
  }
};

struct MatmulOpConverter : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.outputs().size() == 1);
    assert(op.inputs().size() == 2);

    auto encoding = getLinneaTensorEncoding(op.outputs()[0].getType());
    if (!encoding)
      return failure();

    Location loc = op->getLoc();
    Value A = castToTensor(op.inputs()[0], rewriter, loc);
    Value B = castToTensor(op.inputs()[1], rewriter, loc);
    Value C = castToTensor(op.outputs()[0], rewriter, loc);

    Value dest = rewriter
                     .create<linalg::MatmulOp>(loc, TypeRange{C.getType()},
                                               ValueRange{A, B}, C)
                     ->getResult(0);

    rewriter.replaceOpWithNewOp<CastFromBuiltinTensorOp>(op, C.getType(), dest);
    return success();
  }
};

struct ConvertToLoops : public LinneaConvertToLoopsBase<ConvertToLoops> {

  void runOnOperation() override {

    ModuleOp module = getOperation();
    RewritePatternSet patterns(module.getContext());
    patterns.add<FillOpConverter, MatmulOpConverter, InitOpConverter>(
        patterns.getContext());
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createConvertLinneaToLoopsPass() {
  return std::make_unique<ConvertToLoops>();
}
