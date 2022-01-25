//===- LinneaPasses.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_PASSES_H
#define LINNEA_PASSES_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linnea {
std::unique_ptr<OperationPass<FuncOp>> createConvertLinneaToLinalgPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLinneaComprehensivePropertyPropagationPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertLinneaToLoopsPass();
} // namespae linnea
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "Standalone/LinneaPasses.h.inc"

#endif
