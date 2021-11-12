//===- LinneaPasses.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaPasses.h"
#include "Standalone/LinneaExpr.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaTypeConverter.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::linnea;
using namespace mlir::linnea::expr;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

namespace {

/// Type converter
class LinneaTypeConverter : public TypeConverter {
public:
  LinneaTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertMatrixType);
  }
  static Type convertMatrixType(MatrixType type) {
    return RankedTensorType::get(type.getDims(), type.getElementType());
  }
};

// Emit a cascade of matrix ops. Optimization (i.e., matrix-chain
// reordering happen at the symbolic level).
Value matrixCascade(Location loc, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter,
                    TypeConverter *typeConverter) {
  Value left = operands[0];
  Type leftType = typeConverter->convertType(left.getType());
  Value result = nullptr;
  for (size_t i = 1; i < operands.size(); i++) {
    Value right = operands[i];
    Type rightType = typeConverter->convertType(right.getType());
    RankedTensorType dest = RankedTensorType::get(
        {leftType.cast<RankedTensorType>().getShape()[0],
         rightType.cast<RankedTensorType>().getShape()[1]},
        leftType.cast<RankedTensorType>().getElementType());
    Value buffer = rewriter.create<linalg::InitTensorOp>(loc, dest.getShape(),
                                                         dest.getElementType());
    result = rewriter
                 .create<linalg::MatmulOp>(loc, TypeRange{dest},
                                           ValueRange{left, right}, buffer)
                 ->getResult(0);
    left = result;
  }
  return result;
}

class MulOpLowering : public ConversionPattern {
public:
  MulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, MulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() >= 2 && "expect two operands at least");
    Value result =
        matrixCascade(op->getLoc(), operands, rewriter, typeConverter);
    assert(result != nullptr && "must be non null");
    rewriter.replaceOp(op, result);
    return success();
  }
};

class DummyOpLowering : public ConversionPattern {
public:
  DummyOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, DummyOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto input = typeConverter->convertType(op->getOperands()[0].getType())
                     .cast<RankedTensorType>();

    auto result = rewriter.create<linalg::InitTensorOp>(
        op->getLoc(), input.getShape(), input.getElementType());
    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

// Populate patterns
void populateLinneaToLinalgPattern(RewritePatternSet &patterns,
                                   TypeConverter &converter) {
  patterns.add<MulOpLowering>(converter, patterns.getContext());
  patterns.add<DummyOpLowering>(converter, patterns.getContext());
}

struct LowerToLinalg : public LinneaLowerToLinalgBase<LowerToLinalg> {
  void runOnFunction() override {
    ConversionTarget target(getContext());

    RewritePatternSet patterns(&getContext());
    LinneaTypeConverter converter;
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op.getOperandTypes()); });

    populateFuncOpTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    // populateCallOpTypeConversionPattern(patterns, converter);
    // populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateLinneaToLinalgPattern(patterns, converter);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

struct LinneaComprehensivePropertyPropagation
    : public LinneaComprehensivePropertyPropagationBase<
          LinneaComprehensivePropertyPropagation> {
  void runOnOperation() override;
};

static bool isaLinneaTerm(Type t) { return t.isa<mlir::linnea::TermType>(); };

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static ReturnOp getAssumedUniqueReturnOp(FuncOp funcOp) {
  ReturnOp returnOp;
  for (Block &b : funcOp.body()) {
    if (auto candidateOp = dyn_cast<ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

static FunctionType getFunctionType(FuncOp funcOp, TypeRange argumentTypes,
                                    TypeRange resultTypes) {
  return FunctionType::get(funcOp.getContext(), argumentTypes, resultTypes);
}

void LinneaComprehensivePropertyPropagation::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Operation *> toErase;
  WalkResult res = module.walk([&](EquationOp eqOp) -> WalkResult {
    // get terminator. Start building expression terms from the yield op.
    Region &region = eqOp.region();
    Operation *terminator = region.front().getTerminator();
    Value termOperand = terminator->getOperand(0);

    {
      using namespace mlir::linnea::expr;
      ScopedContext ctx;
      ExprBuilder exprBuilder;
      BlockAndValueMapping mapper;
      mapper.map(eqOp.region().getArguments(), eqOp.getOperands());
      Expr *root = exprBuilder.buildLinneaExpr(termOperand);

      // ctx.print();
      // root->walk();
      // root = root->simplify();

      OpBuilder builder(eqOp->getContext());
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(eqOp);
      Value rootVal = exprBuilder.buildIR(eqOp.getLoc(), builder, root, mapper);
      Value resultEqOp = eqOp.getResult();
      resultEqOp.replaceAllUsesWith(rootVal);
      toErase.push_back(eqOp);
    }

    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // adjust at function boundaries.
  // TODO: fix also callee as in comprehensive bufferization pass.
  res = module.walk([](FuncOp funcOp) -> WalkResult {
    if (!llvm::any_of(funcOp.getType().getInputs(), isaLinneaTerm) &&
        !llvm::any_of(funcOp.getType().getResults(), isaLinneaTerm))
      return WalkResult::advance();

    ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    if (!returnOp)
      return funcOp->emitError() << "return op must be available";

    SmallVector<Value> returnValues;
    for (OpOperand &returnOperand : returnOp->getOpOperands()) {
      returnValues.push_back(returnOperand.get());
    }
    ValueRange retValues{returnValues};
    FunctionType funcTypes = getFunctionType(
        funcOp, funcOp.getType().getInputs(), retValues.getTypes());

    Block &front = funcOp.body().front();
    unsigned numArgs = front.getNumArguments();
    for (unsigned idx = 0; idx < numArgs; idx++) {
      auto bbArg = front.getArgument(0);
      auto termType = bbArg.getType().dyn_cast<TermType>();
      if (!termType) {
        front.addArgument(bbArg.getType());
        bbArg.replaceAllUsesWith(front.getArguments().back());
        front.eraseArgument(0);
        continue;
      } else {
        termType.dump();
        assert(0 && "type not supported");
      }
    }
    funcOp.setType(funcTypes);
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    signalPassFailure();
  }

  for (Operation *op : toErase)
    op->erase();

  return;
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createConvertLinneaToLinalgPass() {
  return std::make_unique<LowerToLinalg>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinneaComprehensivePropertyPropagationPass() {
  return std::make_unique<LinneaComprehensivePropertyPropagation>();
}
