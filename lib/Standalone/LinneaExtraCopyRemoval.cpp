//===- LinneaExtraCopyRemoval.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <memory>
#include <utility>

#include "Standalone/LinneaPasses.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-remove-extra-copies"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct LinneaCopyRemoval : public LinneaCopyRemovalBase<LinneaCopyRemoval> {
  void runOnOperation() override;
};

struct CopyOpRemoval : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.source();
    Value target = op.target();
    Operation *defSource = source.getDefiningOp();
    if (!defSource ||
        !(isa<memref::AllocOp>(defSource) || isa<memref::AllocaOp>(defSource)))
      return failure();
    Operation *defTarget = target.getDefiningOp();
    if (!defTarget ||
        !(isa<memref::AllocOp>(defTarget) || isa<memref::AllocaOp>(defTarget)))
      return failure();
    target.replaceAllUsesWith(source);
    rewriter.eraseOp(op);
    return success();
  }
};

void LinneaCopyRemoval::runOnOperation() {
  ModuleOp module = getOperation();
  RewritePatternSet patterns(module.getContext());
  patterns.add<CopyOpRemoval>(patterns.getContext());
  (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  return;
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createLinneaCopyRemoval() {
  return std::make_unique<LinneaCopyRemoval>();
}
