//===- LinneaPasses.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_PASSES_H
#define LINNEA_PASSES_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linnea {
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLinneaToLinalgPass();
std::unique_ptr<OperationPass<ModuleOp>> createLinneaPropertyPropagationPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertLinneaToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLinneaFuncTypeConversion();
std::unique_ptr<OperationPass<ModuleOp>>
createLinneaFinalizeFuncTypeConversion();
std::unique_ptr<OperationPass<ModuleOp>> createLinneaCompilerPipeline();
std::unique_ptr<OperationPass<ModuleOp>> createLinneaCopyRemoval();
} // namespace linnea
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "Standalone/LinneaPasses.h.inc"

#endif
