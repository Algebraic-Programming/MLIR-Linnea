//===- LinneaConvertToLoops.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaPasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-passes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct ConvertToLoops : public LinneaConvertToLoopsBase<ConvertToLoops> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    mlir::PassManager pm(module.getContext());
    pm.addPass(createLinalgComprehensiveModuleBufferizePass());
    (void)pm.run(module);
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createConvertLinneaToLoopsPass() {
  return std::make_unique<ConvertToLoops>();
}
