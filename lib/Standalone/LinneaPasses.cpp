//===- LinneaPasses.cpp -----------------------------------------*- C++ -*-===//
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

#define DEBUG_TYPE "linnea-passes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

/// Type converter
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
static Value emitLinalgMatrix(Location loc, MLIRContext *ctx,
                              ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter,
                              TypeConverter *typeConverter,
                              ResultRange results) {
  assert(operands.size() == 2 && "expect two operands");
  assert(results.size() == 1 && "expect one output");

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

class MulOpLowering : public ConversionPattern {
public:
  MulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, MulOpLow::getOperationName(), 1, ctx) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() >= 2 && "expect two operands at least");
    Value result = emitLinalgMatrix(op->getLoc(), op->getContext(), operands,
                                    rewriter, typeConverter, op->getResults());
    assert(result != nullptr && "must be non null");
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Populate patterns
void populateLinneaToLinalgPattern(RewritePatternSet &patterns,
                                   TypeConverter &converter) {
  patterns.add<MulOpLowering>(converter, patterns.getContext());
}

struct LowerToLinalg : public LinneaLowerToLinalgBase<LowerToLinalg> {
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
    Region &region = eqOp.getBody();
    Operation *terminator = region.front().getTerminator();
    Value termOperand = terminator->getOperand(0);

    {
      using namespace mlir::linnea::expr;
      ScopedContext ctx;
      ExprBuilder exprBuilder;
      Expr *root = exprBuilder.buildLinneaExpr(termOperand);

      // simplify the expression.
      root = root->simplify();
      LLVM_DEBUG(DBGS() << "Simplified expression: \n"; root->walk(););

      OpBuilder builder(eqOp->getContext());
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(eqOp);
      Value rootVal = exprBuilder.buildIR(eqOp->getLoc(), builder, root);
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
        front.addArgument(bbArg.getType(), bbArg.getLoc());
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
