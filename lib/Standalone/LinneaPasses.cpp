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

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linnea;
using namespace mlir::linnea::expr;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-property-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

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

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinneaComprehensivePropertyPropagationPass() {
  return std::make_unique<LinneaComprehensivePropertyPropagation>();
}
