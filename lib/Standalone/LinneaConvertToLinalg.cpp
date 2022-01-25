//===- LinneaConvertToLinalg.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaPasses.h"
#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaExpr.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaTypeConverter.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linnea;
using namespace mlir::linnea::expr;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-convert-to-linalg"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

/// Type converter from MatrixType to RankedTensorType.
class LinneaTypeConverter : public TypeConverter {
public:
  LinneaTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertMatrixType);
  }
  static Type convertMatrixType(MatrixType type) {
    return RankedTensorType::get(type.getDims(), type.getElementType(),
                                 type.getProperty());
  }
};

static inline bool isMLIRFloatType(mlir::Type t) {
  return t.isF16() || t.isF32() || t.isF64();
}

static inline bool isMLIRIntType(mlir::Type t) {
  return t.isInteger(2) || t.isInteger(4) || t.isInteger(8) ||
         t.isInteger(16) || t.isInteger(32) || t.isInteger(64);
}

template <typename FOpTy, typename IOpTy>
static Value buildBinaryOpFromValues(OpBuilder builder, Value left, Value right,
                                     Location loc, Type t) {
  if (isMLIRFloatType(t))
    return builder.create<FOpTy>(loc, left, right);
  else if (isMLIRIntType(t))
    return builder.create<IOpTy>(loc, left, right);
  else
    llvm_unreachable("unsupported type");
}

// Emit linalg matrix op. Optimization (i.e., matrix-chain
// reordering happen at the symbolic level).
static Value emitLinalgMatrix(MulOpHigh op, ValueRange operands,
                              ConversionPatternRewriter &rewriter,
                              TypeConverter *typeConverter,
                              ResultRange results) {
  assert(operands.size() == 2 && "expect two operands");
  assert(results.size() == 1 && "expect one output");
  auto loc = op->getLoc();
  auto ctx = op->getContext();

  Value left = operands[0];
  Value right = operands[1];
  RankedTensorType outputType =
      typeConverter->convertType(results[0].getType()).cast<RankedTensorType>();

  Value buffer = rewriter.create<linalg::InitTensorOp>(
      loc, outputType, ArrayRef<Value>({}),
      rewriter.getI64ArrayAttr(outputType.getShape()));

  // build affine map for matmul.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  auto matMulMap = infer({{m, k}, {k, n}, {m, n}});

  // iterator for matmul.
  llvm::SmallVector<StringRef, 3> iter = {"parallel", "parallel", "reduction"};

  return rewriter
      .create<linalg::GenericOp>(
          loc, TypeRange{outputType}, ValueRange{left, right}, buffer,
          matMulMap, iter,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            assert(args.size() == 3 && "matmul expects 3 args");
            Value mul = buildBinaryOpFromValues<arith::MulFOp, arith::MulIOp>(
                nestedBuilder, args[0], args[1], nestedLoc,
                outputType.getElementType());
            Value add = buildBinaryOpFromValues<arith::AddFOp, arith::AddIOp>(
                nestedBuilder, args[2], mul, nestedLoc,
                outputType.getElementType());
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          })
      ->getResult(0);
}

class MulOpLowering : public OpConversionPattern<MulOpHigh> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MulOpHigh op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.input();
    assert(operands.size() >= 2 && "expect two operands at least");
    Value result = emitLinalgMatrix(op, operands, rewriter, getTypeConverter(),
                                    op->getResults());
    assert(result != nullptr && "must be non null");
    rewriter.replaceOp(op, result);
    return success();
  }
};

class FillOpLowering : public OpConversionPattern<FillOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType =
        getTypeConverter()->convertType(adaptor.output().getType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        op, newResultType, adaptor.value(), adaptor.output());
    return success();
  }
};

// Populate patterns
void populateLinneaToLinalgPattern(RewritePatternSet &patterns,
                                   TypeConverter &converter) {
  patterns.add<MulOpLowering>(converter, patterns.getContext());
  patterns.add<FillOpLowering>(converter, patterns.getContext());
}

struct ConvertToLinalg : public LinneaConvertToLinalgBase<ConvertToLinalg> {
  void runOnFunction() override {
    ConversionTarget target(getContext());

    RewritePatternSet patterns(&getContext());
    LinneaTypeConverter converter;
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();

    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op.getOperandTypes()); });

    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateLinneaToLinalgPattern(patterns, converter);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::linnea::createConvertLinneaToLinalgPass() {
  return std::make_unique<ConvertToLinalg>();
}
