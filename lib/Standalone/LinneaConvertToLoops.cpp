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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
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

static LinneaMatrixEncodingAttr getLinneaTensorEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<LinneaMatrixEncodingAttr>();
  return nullptr;
}

struct LoopNestInfo {
  SmallVector<Value> ubs;
  SmallVector<Value> lbs;
  SmallVector<Value> steps;
};

LoopNestInfo getLoopNestInfo(linalg::FillOp op) {}

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

// TODO: fix diagonal i <= j
static void buildLoopNestTriangularImpl(PatternRewriter &rewriter,
                                        linalg::FillOp op, Value destMemRef) {
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

  auto currInsertionPoint = rewriter.getInsertionPoint();
  auto currInsertionBlock = rewriter.getInsertionBlock();
  scf::ForOp outerForOp =
      rewriter.create<scf::ForOp>(loc, lbs[0], ubs[0], steps[0]);
  rewriter.setInsertionPointToStart(&outerForOp.getLoopBody().front());
  scf::ForOp innerForOp = rewriter.create<scf::ForOp>(
      loc, lbs[1], outerForOp.getInductionVar(), steps[1]);
  rewriter.setInsertionPointToStart(&innerForOp.getLoopBody().front());
  SmallVector<Value> ivs = {outerForOp.getInductionVar(),
                            innerForOp.getInductionVar()};
  rewriter.create<memref::StoreOp>(loc, op.value(), destMemRef, ivs);
  rewriter.setInsertionPoint(currInsertionBlock, currInsertionPoint);
}

static void buildLoopNest(PatternRewriter &rewriter, linalg::FillOp op,
                          Value destMemRef) {
  ArrayRef<LinneaMatrixEncodingAttr::MatrixProperty> encoding =
      getLinneaTensorEncoding(op.output().getType()).getEncoding();
  if (encoding.size() == 1 &&
      encoding[0] ==
          LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular) {
    buildLoopNestTriangularImpl(rewriter, op, destMemRef);
    return;
  }
  buildLoopNestImpl(rewriter, op, destMemRef);
}

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
    MemRefType memref =
        MemRefType::get(outputTensor.getShape(), outputTensor.getElementType());
    Location loc = op->getLoc();
    Type castedLinneaTensorType = RankedTensorType::get(
        outputTensor.getShape(), outputTensor.getElementType());
    Value castedLinneaTensor = rewriter.create<CastToBuiltinTensorOp>(
        loc, castedLinneaTensorType, op.output());
    Value dest = rewriter.create<bufferization::ToMemrefOp>(loc, memref,
                                                            castedLinneaTensor);

    buildLoopNest(rewriter, op, dest);
    /*
        SmallVector<Value> ubs;
        Value dim1 = rewriter.create<arith::ConstantIndexOp>(
            loc, outputTensor.getShape()[0]);
        Value dim2 = rewriter.create<arith::ConstantIndexOp>(
            loc, outputTensor.getShape()[1]);
        ubs = {dim1, dim2};
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        SmallVector<Value> lbs = {zero, zero};
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        SmallVector<Value> steps = {one, one};

        auto loopNest = scf::buildLoopNest(
            rewriter, loc, lbs, ubs, steps,
            [&](OpBuilder &b, Location loc, ValueRange localIvs) {
              b.create<memref::StoreOp>(loc, op.value(), dest, localIvs);
            });
    */
    Value ret = rewriter.create<bufferization::ToTensorOp>(
        loc, castedLinneaTensorType, dest);
    rewriter.replaceOpWithNewOp<CastFromBuiltinTensorOp>(
        op, op.output().getType(), ret);
    return success();

    return failure();
  }
};

struct ConvertToLoops : public LinneaConvertToLoopsBase<ConvertToLoops> {

  /*
    void getDependentDialects(DialectRegistry &registry) const override {
      registry
          .insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, tensor::TensorDialect,
                  vector::VectorDialect, scf::SCFDialect,
                  arith::ArithmeticDialect, StandardOpsDialect,
    AffineDialect>();
      // register linalg interface.
      linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
    }
  */

  void runOnOperation() override {

    ModuleOp module = getOperation();
    /*
        mlir::PassManager pm(module.getContext());
        pm.addPass(createLinalgComprehensiveModuleBufferizePass());
        (void)pm.run(module);
    */
    RewritePatternSet patterns(module.getContext());
    patterns.add<FillOpConverter>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createConvertLinneaToLoopsPass() {
  return std::make_unique<ConvertToLoops>();
}
