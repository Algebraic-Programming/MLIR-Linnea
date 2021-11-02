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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::linnea;
using namespace mlir::linnea::expr;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

namespace {

SmallVector<int64_t, 2> getOutputShape(Value x, Value y) {
  RankedTensorType xType = x.getType().cast<RankedTensorType>();
  RankedTensorType yType = y.getType().cast<RankedTensorType>();
  assert(xType && "must be valid type");
  assert(yType && "must be valid type");
  auto shapeX = xType.getShape();
  auto shapeY = yType.getShape();
  assert(shapeX.size() == 2 && "expect 2d tensor");
  assert(shapeY.size() == 2 && "expect 2d tensor");
  return {shapeX[0], shapeY[1]};
}

Type getElementType(Value v) {
  RankedTensorType vType = v.getType().cast<RankedTensorType>();
  assert(vType && "must be a valid type");
  return vType.getElementType();
}

SmallVector<long, 8> getPVector(ArrayRef<Value> &operands) {
  SmallVector<long, 8> pVector;
  for (Value value : operands) {
    RankedTensorType tensorType =
        value.getType().dyn_cast_or_null<RankedTensorType>();
    if (!tensorType || !tensorType.hasStaticShape())
      return {};
    auto shape = tensorType.getShape();
    if (!pVector.size()) {
      pVector.push_back(shape[0]);
      pVector.push_back(shape[1]);
    } else {
      pVector.push_back(shape[1]);
    }
  }
  return pVector;
}

void printOptimalParens(const std::vector<std::vector<long>> &s, size_t i,
                        size_t j, ArrayRef<Value> operands) {
  if (i == j) {
    llvm::errs() << " ";
    operands[i - 1].getType().dump();
    llvm::errs() << "  ";
  } else {
    llvm::errs() << "(";
    printOptimalParens(s, i, s[i][j], operands);
    printOptimalParens(s, s[i][j] + 1, j, operands);
    llvm::errs() << ")";
  }
}

std::vector<std::vector<long>> getOptimalSplit(ArrayRef<Value> &operands) {

  SmallVector<long, 8> pVector = getPVector(operands);
  const size_t n = pVector.size();
  std::vector<std::vector<long>> m(
      n, std::vector<long>(n, std::numeric_limits<long>::max()));
  std::vector<std::vector<long>> s(
      n, std::vector<long>(n, std::numeric_limits<long>::max()));

  for (size_t i = 0; i < n; i++)
    m[i][i] = 0;

  size_t j = 0;
  long q = 0;
  for (size_t l = 2; l < n; l++) {
    for (size_t i = 1; i < n - l + 1; i++) {
      j = i + l - 1;
      m[i][j] = std::numeric_limits<long>::max();
      for (size_t k = i; k <= j - 1; k++) {
        q = m[i][k] + m[k + 1][j] + pVector[i - 1] * pVector[k] * pVector[j];
        if (q < m[i][j]) {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }
  /*
    llvm::errs() << "\n\n-----s------\n";
    int rows = s.size();
    int cols = s[0].size();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (s[i][j] == std::numeric_limits<long>::max())
          llvm::errs() << "- ";
        else
          llvm::errs() << s[i][j] << " ";
      }
      llvm::errs() << "\n";
    }

    llvm::errs() << "-----m------\n";
    rows = m.size();
    cols = m[0].size();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (m[i][j] == std::numeric_limits<long>::max())
          llvm::errs() << "- ";
        else
          llvm::errs() << m[i][j] << " ";
      }
      llvm::errs() << "\n";
    }
    printOptimalParens(s, 1, operands.size(), operands);
    llvm::errs() << "\n\n";
  */
  return s;
}

/// Check preconditions. The chain of multiplication
/// should be compatible.
LogicalResult checkPreconditions(ArrayRef<Value> operands) {
  ArrayRef<int64_t> left =
      operands[0].getType().cast<RankedTensorType>().getShape();
  Type elementTypeLeft =
      operands[0].getType().cast<RankedTensorType>().getElementType();
  for (size_t i = 1; i < operands.size(); i++) {
    ArrayRef<int64_t> right =
        operands[i].getType().cast<RankedTensorType>().getShape();
    Type elementTypeRight =
        operands[i].getType().cast<RankedTensorType>().getElementType();
    if (elementTypeLeft != elementTypeRight)
      return failure();
    assert(left.size() == 2 && "expect 2d tensor");
    assert(right.size() == 2 && "expect 2d tensor");
    if (left[1] != right[0])
      return failure();
    left = ArrayRef<int64_t>{left[0], right[1]};
  }
  return success();
}

/// Rebuild chain using best split.
Value buildChain(Location loc, ArrayRef<Value> operands,
                 std::vector<std::vector<long>> &split, int i, int j,
                 ConversionPatternRewriter &rewriter) {
  if (i < j) {
    Value x = buildChain(loc, operands, split, i, split[i][j], rewriter);
    Value y = buildChain(loc, operands, split, split[i][j] + 1, j, rewriter);
    SmallVector<int64_t> outShape = getOutputShape(x, y);
    Value buffer =
        rewriter.create<linalg::InitTensorOp>(loc, outShape, getElementType(x));
    return rewriter
        .create<linalg::MatmulOp>(loc, TypeRange{buffer.getType()},
                                  ValueRange{x, y}, buffer)
        ->getResult(0);
  } else {
    return operands[i - 1];
  }
}

/// Opt with matrix-chain reordering.
Value matrixChainOpt(Location loc, ArrayRef<Value> operands,
                     ConversionPatternRewriter &rewriter) {
  std::vector<std::vector<long>> optimalSplit = getOptimalSplit(operands);
  return buildChain(loc, operands, optimalSplit, 1, operands.size(), rewriter);
}

/// Just emit a cascade of matrix ops.
Value matrixCascade(Location loc, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) {
  Value left = operands[0];
  Value result = nullptr;
  for (size_t i = 1; i < operands.size(); i++) {
    Value right = operands[i];
    Value leftDim = rewriter.createOrFold<tensor::DimOp>(loc, left, 0);
    Value rightDim = rewriter.createOrFold<tensor::DimOp>(loc, right, 1);
    SmallVector<Value, 2> outShape{leftDim, rightDim};
    Value buffer = rewriter.create<linalg::InitTensorOp>(loc, outShape,
                                                         getElementType(left));
    result = rewriter
                 .create<linalg::MatmulOp>(loc, TypeRange{buffer.getType()},
                                           ValueRange{left, right}, buffer)
                 ->getResult(0);
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
    typeConverter->convertType(op->getOperands()[0].getType());
    // if (failed(checkPreconditions(operands)))
    //  return failure();
    // Value result = nullptr;
    // if (operands.size() > 2 && llvm::all_of(operands, [](Value v) {
    //      return v.getType().cast<RankedTensorType>().hasStaticShape();
    //    }))
    //  result = matrixChainOpt(op->getLoc(), operands, rewriter);
    // else
    //  result = matrixCascade(op->getLoc(), operands, rewriter);
    //
    // assert(result != nullptr && "must be non null");
    // rewriter.replaceOp(op, result);
    return failure();
  }
};

/// Populate
void populateLinneaToLinalgPattern(TypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<MulOpLowering>(converter, patterns.getContext());
}

Type convertMatrixType(MatrixType type) {
  assert(0 && "conversion not available");
  return Type();
}

void pupulateLinneaTypeToTensorTypePattern(LinneaTypeConverter &converter) {
  converter.addConversion(
      [](MatrixType type) { return convertMatrixType(type); });
}

struct LowerToLinalg : public LinneaLowerToLinalgBase<LowerToLinalg> {
  void runOnFunction() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, memref::MemRefDialect,
                           StandardOpsDialect, BuiltinDialect,
                           mlir::tensor::TensorDialect>();

    RewritePatternSet patterns(&getContext());
    LinneaTypeConverter converter;
    pupulateLinneaTypeToTensorTypePattern(converter);
    populateLinneaToLinalgPattern(converter, patterns);
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

struct LinneaComprehensiveTensorMaterialization
    : public LinneaComprehensiveTensorMaterializationBase<
          LinneaComprehensiveTensorMaterialization> {
  void runOnOperation() override;
};

// static bool isaMatrixType(Type t) { return t.isa<mlir::linnea::MatrixType>();
// };

LogicalResult inPlaceUpdateFuncOp(FuncOp funcOp) {
  /*
    SmallVector<Operation *> ops;
    funcOp.walk([&](Operation *op) {
      if (none_of(op->getOperandTypes(), isaMatrixType) &&
          none_of(op->getResultTypes(), isaMatrixType))
        return;
      ops.push_back(op);
    });

    if (!ops.size())
      return success();

    for (Operation *op : reverse(ops)) {
      if (isa<ReturnOp>(op))
        continue;
      OpBuilder builder(op);
      OpBuilder::InsertionGuard guard(builder);
      SmallVector<Type> operandTypes;
      for (OpOperand operand : op->getOperands())


    }
  */
  return success();
}

LogicalResult boundaryUpdateFuncOp(FuncOp funcOp) { return success(); }

void LinneaComprehensiveTensorMaterialization::runOnOperation() {
  ModuleOp module = getOperation();
  WalkResult res = module.walk([](FuncOp funcOp) -> WalkResult {
    if (failed(inPlaceUpdateFuncOp(funcOp)))
      return WalkResult::interrupt();
    if (failed(boundaryUpdateFuncOp(funcOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (res.wasInterrupted())
    signalPassFailure();
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

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinneaComprehensiveTensorMaterializationPass() {
  return std::make_unique<LinneaComprehensiveTensorMaterialization>();
}
