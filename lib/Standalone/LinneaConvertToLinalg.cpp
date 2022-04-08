//===- LinneaConvertToLinalg.cpp ---------------------------------*- C++-*-===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linnea;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-convert-to-linalg"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// Return the unique ReturnOp that terminates `funcOp`.
// Return nullptr if there is no such unique ReturnOp.
static inline func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

// Materialize a buffer of type 'outputType'. Since we lower initTensorOp
// to an alloc we also emit a delloc. The allocated buffer is filled with
// zeros.
static Value emitAllocAndDealloc(MulOpLow op, RankedTensorType outputType,
                                 ConversionPatternRewriter &rewriter,
                                 StringAttr semirings = nullptr) {
  Location loc = op->getLoc();
  // Materialize result.
  Value buffer = rewriter.create<linalg::InitTensorOp>(
      loc, outputType, ArrayRef<Value>({}),
      rewriter.getI64ArrayAttr(outputType.getShape()));

  // Fill result buffer with 0.
  Attribute resultZeroAttr = rewriter.getZeroAttr(outputType.getElementType());
  if (semirings) {
    Type elementType = outputType.getElementType();
    if (semirings.str().compare("min-plus") == 0) {
      assert(isMLIRFloatType(elementType) &&
             "expect float type with 'min-plus' semirings");
      resultZeroAttr = rewriter.getFloatAttr(
          elementType,
          APFloat::getInf(elementType.cast<FloatType>().getFloatSemantics(),
                          /*Negative*/ false));
    } else if (semirings.str().compare("max-plus") == 0) {
      assert(isMLIRFloatType(elementType) &&
             "expect float type with 'max-plus' semirings");
      resultZeroAttr = rewriter.getFloatAttr(
          elementType,
          APFloat::getInf(elementType.cast<FloatType>().getFloatSemantics(),
                          /*Negative*/ true));
    }
  }
  Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
  buffer = rewriter.create<linalg::FillOp>(loc, zero, buffer).getResult(0);

  // insert a dealloc before exiting the function.
  auto currentInsertionPoint = rewriter.getInsertionPoint();
  auto *currentInsertionBlock = rewriter.getInsertionBlock();
  func::ReturnOp ret =
      getAssumedUniqueReturnOp(op->getParentOfType<func::FuncOp>());
  assert(ret && "assume a return op");

  rewriter.setInsertionPoint(ret);
  rewriter.create<linnea::DeallocOp>(ret->getLoc(), buffer);

  // restore insertion point.
  rewriter.setInsertionPoint(currentInsertionBlock, currentInsertionPoint);

  return buffer;
}

// Emit linalg matrix op. Optimization (i.e., matrix-chain
// reordering happen at the symbolic level).
static Value emitLinalgMatrix(MulOpLow op, ValueRange operands,
                              ConversionPatternRewriter &rewriter,
                              TypeConverter *typeConverter,
                              ResultRange results) {
  Value left = operands[0];
  Value right = operands[1];
  RankedTensorType outputType =
      typeConverter->convertType(results[0].getType()).cast<RankedTensorType>();

  Value buffer = emitAllocAndDealloc(op, outputType, rewriter);

  return rewriter
      .create<linalg::MatmulOp>(op->getLoc(), TypeRange{buffer.getType()},
                                ValueRange{left, right}, buffer)
      ->getResult(0);
}

static Value buildMatrixBody(StringAttr semirings, ValueRange operands,
                             OpBuilder &builder, Location loc) {
  if ((semirings.str().compare("real-arith") == 0) ||
      (semirings.str().compare("integer-arith") == 0)) {
    assert(0 && "not supported");
  } else if (semirings.str().compare("min-plus") == 0) {
    Type elementType = operands[2].getType();
    assert(isMLIRFloatType(elementType) &&
           "expect float type with 'min-plus' semirings");
    Value add = buildBinaryOpFromValues<arith::MinFOp, arith::MinSIOp>(
        builder, operands[0], operands[1], loc, elementType);
    Value mul = buildBinaryOpFromValues<arith::AddFOp, arith::AddIOp>(
        builder, operands[2], add, loc, elementType);
    return mul;
  } else if (semirings.str().compare("max-plus") == 0) {
    Type elementType = operands[2].getType();
    assert(isMLIRFloatType(elementType) &&
           "expect float type with 'max-plus' semirings");
    Value add = buildBinaryOpFromValues<arith::MaxFOp, arith::MaxSIOp>(
        builder, operands[0], operands[1], loc, elementType);
    Value mul = buildBinaryOpFromValues<arith::AddFOp, arith::AddIOp>(
        builder, operands[2], add, loc, elementType);
    return mul;
  }
  llvm_unreachable("semirings not supported");
}

// Emit a linalg matrix based on the semirings attribute.
static Value emitLinalgMatrixWithSem(MulOpLow op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter,
                                     TypeConverter *typeConverter,
                                     ResultRange results,
                                     StringAttr semirings) {
  Value left = operands[0];
  Value right = operands[1];
  RankedTensorType outputType =
      typeConverter->convertType(results[0].getType()).cast<RankedTensorType>();

  Value buffer = emitAllocAndDealloc(op, outputType, rewriter, semirings);

  // build affine maps for the mul operation.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr i, j, k;
  bindDims(op->getContext(), i, j, k);
  auto mulMap = infer({{i, k}, {k, j}, {i, j}});
  // iterators.
  SmallVector<StringRef, 3> iter = {"parallel", "parallel", "reduction"};

  Value result =
      rewriter
          .create<linalg::GenericOp>(
              op->getLoc(), TypeRange{buffer.getType()},
              ValueRange{left, right}, buffer, mulMap, iter,
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange args) {
                assert(args.size() == 3 && "expect 3 args");
                Value mul =
                    buildMatrixBody(semirings, args, nestedBuilder, nestedLoc);
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, mul);
              })
          ->getResult(0);

  return result;
}

static bool isRealOrArith(StringAttr semirings) {
  if ((semirings.str().compare("real-arith") == 0) ||
      (semirings.str().compare("integer-arith") == 0))
    return true;
  return false;
}

/// Linnea conversion rule for MulOpLow.
class MulOpConverter : public OpConversionPattern<MulOpLow> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MulOpLow op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr attr = op->getAttr("semirings").dyn_cast_or_null<StringAttr>();
    assert(attr);
    ValueRange operands = {adaptor.left(), adaptor.right()};
    // here we emit linalg.matmul for real and integer. On the long run
    // we will use only linalg.generic.
    Value result = (isRealOrArith(attr) == true)
                       ? emitLinalgMatrix(op, operands, rewriter,
                                          getTypeConverter(), op->getResults())
                       : emitLinalgMatrixWithSem(op, operands, rewriter,
                                                 getTypeConverter(),
                                                 op->getResults(), attr);
    assert(result != nullptr && "must be non null");
    rewriter.replaceOp(op, result);
    return success();
  }
};

static Value emitLinalgAdd(AddOpLow op, ValueRange operands,
                           ConversionPatternRewriter &rewriter,
                           TypeConverter *typeConverter, ResultRange results) {
  assert(operands.size() == 2 && "expect two operands");
  assert(results.size() == 1 && "expect one input");
  Location loc = op->getLoc();

  Value left = operands[0];
  Value right = operands[1];
  RankedTensorType outputType =
      typeConverter->convertType(results[0].getType()).cast<RankedTensorType>();

  // Materialize result.
  Value buffer = rewriter.create<linalg::InitTensorOp>(
      loc, outputType, ArrayRef<Value>({}),
      rewriter.getI64ArrayAttr(outputType.getShape()));

  // Fill result buffer with 0.
  Attribute resultZeroAttr = rewriter.getZeroAttr(outputType.getElementType());
  Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
  buffer = rewriter.create<linalg::FillOp>(loc, zero, buffer).getResult(0);

  // build affine map for add operation.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n;
  bindDims(op->getContext(), m, n);
  auto addMap = infer({{m, n}, {m, n}, {m, n}});

  // iterator for add operation.
  llvm::SmallVector<StringRef, 3> iter = {"parallel", "parallel"};

  return rewriter
      .create<linalg::GenericOp>(
          loc, TypeRange{buffer.getType()}, ValueRange{left, right}, buffer,
          addMap, iter,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value add = buildBinaryOpFromValues<arith::AddFOp, arith::AddIOp>(
                nestedBuilder, args[0], args[1], nestedLoc,
                outputType.getElementType());
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          })
      ->getResult(0);
}

/// Linnea conversion rule for AddOpLow.
class AddOpConverter : public OpConversionPattern<AddOpLow> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOpLow op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = {adaptor.left(), adaptor.right()};
    Value result = emitLinalgAdd(op, operands, rewriter, getTypeConverter(),
                                 op->getResults());
    assert(result != nullptr && "must be non null");
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Linnea conversion rule for FillOp.
class FillOpConverter : public OpConversionPattern<FillOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.output().getType();
    auto encoding = getLinneaMatrixEncoding(resType);
    if (!encoding)
      return failure();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, adaptor.value(),
                                                adaptor.output());
    return success();
  }
};

/// Linnea conversion rule for AllocOp.
class AllocOpConverter : public OpConversionPattern<AllocOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    auto encoding = getLinneaMatrixEncoding(resType);
    if (!encoding)
      return failure();

    auto linneaType = resType.cast<MatrixType>();
    RankedTensorType builtinTensorWithProperty =
        RankedTensorType::get(linneaType.getDims(), linneaType.getElementType(),
                              linneaType.getProperty());

    rewriter.replaceOpWithNewOp<linalg::InitTensorOp>(
        op, builtinTensorWithProperty, ArrayRef<Value>{},
        rewriter.getI64ArrayAttr(builtinTensorWithProperty.getShape()));
    return success();
  }
};

/// Linnea conversion rule for DeallocOp.
class DeallocOpConverter : public OpConversionPattern<DeallocOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inputType = op.input().getType();
    auto encoding = getLinneaMatrixEncoding(inputType);
    if (!encoding)
      return failure();

    auto linneaType = inputType.cast<MatrixType>();
    RankedTensorType castedTensorType =
        RankedTensorType::get(linneaType.getDims(), linneaType.getElementType(),
                              linneaType.getProperty());
    Value castedToTensor = rewriter.create<ToBuiltinTensorOp>(
        op->getLoc(), castedTensorType, op.input());
    rewriter.updateRootInPlace(op, [&] { op->setOperands(castedToTensor); });
    return success();
  }
};

/// Linnea conversion for PrintOp.
class PrintOpConverter : public OpConversionPattern<PrintOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.source().getType();
    auto encoding = getLinneaMatrixEncoding(resType);
    if (!encoding)
      return failure();
    Location loc = op->getLoc();
    auto linneaType = resType.cast<MatrixType>();
    RankedTensorType castedTensorType =
        RankedTensorType::get(linneaType.getDims(), linneaType.getElementType(),
                              linneaType.getProperty());
    Value castedToTensor =
        rewriter.create<ToBuiltinTensorOp>(loc, castedTensorType, op.source());
    castedTensorType = RankedTensorType::get(linneaType.getDims(),
                                             linneaType.getElementType());
    castedToTensor = rewriter.create<CastToBuiltinTensorOp>(
        loc, castedTensorType, castedToTensor);
    MemRefType memrefType =
        MemRefType::get(linneaType.getDims(), linneaType.getElementType());
    Value memrefVal = rewriter.create<bufferization::ToMemrefOp>(
        loc, memrefType, castedToTensor);
    // use the print in the vector dialect.
    VectorType vecType =
        VectorType::get(linneaType.getDims(), linneaType.getElementType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // TODO: fix me only f32 or i32.
    Value minusOne = nullptr;
    if (isMLIRFloatType(linneaType.getElementType()))
      minusOne =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(-1));
    else
      minusOne = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(-1));

    Value vecRead = rewriter.create<vector::TransferReadOp>(
        loc, vecType, memrefVal, ArrayRef<Value>{zero, zero}, minusOne);
    rewriter.replaceOpWithNewOp<vector::PrintOp>(op, vecRead);
    return success();
  }
};

// Populate patterns.
void populateLinneaToLinalgPattern(RewritePatternSet &patterns,
                                   TypeConverter &converter) {
  patterns.add<FillOpConverter, MulOpConverter, AllocOpConverter,
               DeallocOpConverter, PrintOpConverter, AddOpConverter>(
      converter, patterns.getContext());
}

static void setupTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter) {
  target.addLegalOp<ToBuiltinTensorOp>();
  typeConverter.addConversion([](MatrixType type) -> RankedTensorType {
    return RankedTensorType::get(type.getDims(), type.getElementType(),
                                 type.getProperty());
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, TensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1 && "expects one input only");
    assert(inputs[0].getType().isa<MatrixType>() && "must be a MatrixType");
    return builder.create<ToBuiltinTensorOp>(loc, type, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder, Type type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1 && "expects one input only");
    assert(inputs[0].getType().isa<TensorType>() && "must be a TensorType");
    return builder.create<FromBuiltinTensorOp>(loc, type, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

// In a finalizing conversion, we know that all of the source types have been
// converted to the destination types, so the materialization becomes an
// identity.
template <typename OpTy>
class FinalizeMaterialization : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

template <typename OpTy>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  target.addIllegalOp<OpTy>();
  patterns.add<FinalizeMaterialization<OpTy>>(typeConverter,
                                              patterns.getContext());
}

template <typename OpTy, typename OpTy2, typename... OpTys>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  setupFinalization<OpTy>(target, patterns, typeConverter);
  setupFinalization<OpTy2, OpTys...>(target, patterns, typeConverter);
}

struct ConvertToLinalg : public LinneaConvertToLinalgBase<ConvertToLinalg> {
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    TypeConverter typeConverter;
    ConversionTarget target(getContext());

    typeConverter.addConversion([](Type type) { return type; });
    setupTypeConversion(target, typeConverter);

    target.addLegalDialect<linalg::LinalgDialect>();

    setupFinalization<ToBuiltinTensorOp, FromBuiltinTensorOp>(target, patterns,
                                                              typeConverter);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    populateLinneaToLinalgPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::linnea::createConvertLinneaToLinalgPass() {
  return std::make_unique<ConvertToLinalg>();
}
