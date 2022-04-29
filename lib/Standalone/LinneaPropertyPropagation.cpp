//===- LinneaPropertyPropagation.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaExpr.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <stack>

using namespace mlir;
using namespace mlir::linnea;
using namespace mlir::func;
using namespace mlir::linnea::expr;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-property-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct LinneaPropertyPropagation
    : public LinneaPropertyPropagationBase<LinneaPropertyPropagation> {
  void runOnOperation() override;
};

static bool isaLinneaTerm(Type t) { return t.isa<mlir::linnea::TermType>(); };

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static func::ReturnOp getAssumedUniqueReturnOp(FuncOp funcOp) {
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

static FunctionType getFunctionType(FuncOp funcOp, TypeRange argumentTypes,
                                    TypeRange resultTypes) {
  return FunctionType::get(funcOp.getContext(), argumentTypes, resultTypes);
}

void LinneaPropertyPropagation::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Operation *> toErase;

  std::stack<EquationOp> frontier;
  module.walk([&](EquationOp eqOp) { frontier.push(eqOp); });
  std::stack<std::pair<EquationOp, Expr *>> simplifiedExpressions;

  if (frontier.empty())
    return;

  {
    using namespace mlir::linnea::expr;
    ScopedContext ctx;
    ExprBuilder exprBuilder;

    while (!frontier.empty()) {
      EquationOp eqOp = frontier.top();
      frontier.pop();

      if (exprBuilder.isAlreadyVisited(eqOp)) {
        toErase.push_back(eqOp);
        continue;
      }

      Block &block = eqOp.getBody();
      Operation *terminator = block.getTerminator();
      Value termOperand = terminator->getOperand(0);
      Expr *root =
          exprBuilder.buildLinneaExpr(termOperand, eqOp.getOperation());

      // root->walk();
      LLVM_DEBUG(DBGS() << "Before simplify ---> \n"; root->walk(););
      root = root->simplify(SymbolicOpt);
      LLVM_DEBUG(DBGS() << "After simplify ---> \n"; root->walk(););
      // root->walk();
      simplifiedExpressions.push({eqOp, root});
      toErase.push_back(eqOp);
    }

    while (!simplifiedExpressions.empty()) {
      auto simplifiedExpr = simplifiedExpressions.top();
      simplifiedExpressions.pop();
      EquationOp eqOp = simplifiedExpr.first;
      OpBuilder builder(eqOp->getContext());
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(eqOp);
      Value rootVal =
          exprBuilder.buildIR(eqOp->getLoc(), builder, simplifiedExpr.second);
      Value resultEqOp = eqOp.getResult();
      resultEqOp.replaceAllUsesWith(rootVal);
    }

  } // scoped context.

  // adjust at function boundaries.
  // TODO: fix also callee as in comprehensive bufferization pass.
  WalkResult res = module.walk([](FuncOp funcOp) -> WalkResult {
    if (!llvm::any_of(funcOp.getFunctionType().getInputs(), isaLinneaTerm) &&
        !llvm::any_of(funcOp.getFunctionType().getResults(), isaLinneaTerm))
      return WalkResult::advance();

    func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    if (!returnOp)
      return funcOp->emitError() << "return op must be available";

    SmallVector<Value> returnValues;
    for (OpOperand &returnOperand : returnOp->getOpOperands()) {
      returnValues.push_back(returnOperand.get());
    }
    ValueRange retValues{returnValues};
    FunctionType funcTypes = getFunctionType(
        funcOp, funcOp.getFunctionType().getInputs(), retValues.getTypes());

    Block &front = funcOp.getBody().front();
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
mlir::linnea::createLinneaPropertyPropagationPass() {
  return std::make_unique<LinneaPropertyPropagation>();
}
