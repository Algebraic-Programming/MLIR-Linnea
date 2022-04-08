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

/// Converter for linalg::InitTensorOp.
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
    Value memref = rewriter.create<memref::AllocOp>(loc, memTp);
    RankedTensorType builtinTensor = RankedTensorType::get(
        outputTensor.getShape(), outputTensor.getElementType());
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, builtinTensor,
                                                           memref);
    return success();
  }
};

/// Converter for linnea::DeallocOp.
struct DeallocOpConverter : public OpRewritePattern<linnea::DeallocOp> {
public:
  using OpRewritePattern<linnea::DeallocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linnea::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    Type input = op.input().getType();
    auto encoding = getLinneaTensorEncoding(input);
    if (!encoding)
      return failure();

    Location loc = op->getLoc();
    Value memref = castToMemRef(op.input(), rewriter, loc);
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, memref);
    return success();
  }
};

/// Converter for linalg::FillOp.
struct FillOpConverter : public OpRewritePattern<linalg::FillOp> {
public:
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
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

/// Converter for a linalg.matmulOp.
// TODO: duplicated code see AddOpConverter. Also it just introduce
// some cast. Do we really need it?
struct MatmulOpConverter : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.outputs().size() == 1 && "expect single ouptut");
    assert(op.inputs().size() == 2 && "expect two inputs");

    auto encoding = getLinneaTensorEncoding(op.outputs()[0].getType());
    if (!encoding)
      return failure();

    Location loc = op->getLoc();
    Value a = castToTensor(op.inputs()[0], rewriter, loc);
    Value b = castToTensor(op.inputs()[1], rewriter, loc);
    Value c = castToTensor(op.outputs()[0], rewriter, loc);

    Value dest = rewriter
                     .create<linalg::MatmulOp>(loc, TypeRange{c.getType()},
                                               ValueRange{a, b}, c)
                     ->getResult(0);

    rewriter.replaceOpWithNewOp<CastFromBuiltinTensorOp>(op, c.getType(), dest);
    return success();
  }
};

/// Converter for linalg::GenericOp (coming from a linnea add).
// TODO: again too much code duplication.
struct AddOpConverter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.outputs().size() == 1 && "expect single output");
    assert(op.inputs().size() == 2 && "expect two inputs");

    auto encoding = getLinneaTensorEncoding(op.outputs()[0].getType());
    if (!encoding)
      return failure();

    Location loc = op->getLoc();
    Value a = castToTensor(op.inputs()[0], rewriter, loc);
    Value b = castToTensor(op.inputs()[1], rewriter, loc);
    Value c = castToTensor(op.outputs()[0], rewriter, loc);

    // build affine map for add operation.
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr m, n;
    bindDims(op->getContext(), m, n);
    auto addMap = infer({{m, n}, {m, n}, {m, n}});

    // iterator for add operation.
    SmallVector<StringRef, 2> iter = {"parallel", "parallel"};

    Value dest =
        rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{c.getType()}, ValueRange{a, b}, c, addMap, iter,
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  Value add =
                      buildBinaryOpFromValues<arith::AddFOp, arith::AddIOp>(
                          nestedBuilder, args[0], args[1], nestedLoc,
                          args[2].getType());
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
                })
            ->getResult(0);
    rewriter.replaceOpWithNewOp<CastFromBuiltinTensorOp>(op, c.getType(), dest);
    return success();
  }
};

struct ConvertToLoops : public LinneaConvertToLoopsBase<ConvertToLoops> {

  void runOnOperation() override {

    ModuleOp module = getOperation();
    RewritePatternSet patterns(module.getContext());
    patterns.add<FillOpConverter, MatmulOpConverter, AddOpConverter,
                 InitOpConverter, DeallocOpConverter>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createConvertLinneaToLoopsPass() {
  return std::make_unique<ConvertToLoops>();
}
