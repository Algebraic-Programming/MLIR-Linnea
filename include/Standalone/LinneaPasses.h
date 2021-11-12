//===- LinneaPasses.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_PASSES_H
#define LINNEA_PASSES_H

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// namespace linnea {
std::unique_ptr<OperationPass<FuncOp>> createConvertLinneaToLinalgPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLinneaComprehensivePropertyPropagationPass();
//} // end linnea
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "Standalone/LinneaPasses.h.inc"

#endif
