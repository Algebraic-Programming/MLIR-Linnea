//===- LinneaPasses.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_PASSES_H
#define LINNEA_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir {
namespace vector {
class VectorDialect;
} // namespace vector
} // namespace mlir

namespace mlir {
namespace linalg {
class LinalgDialect;
} // namespace linalg
} // namespace mlir

namespace mlir {
namespace scf {
class SCFDialect;
} // namespace scf
} // namespace mlir

namespace mlir {
namespace memref {
class MemRefDialect;
} // namespace memref
} // namespace mlir

namespace mlir {
namespace arith {
class ArithmeticDialect;
} // namespace arith
} // namespace mlir

namespace mlir {
namespace bufferization {
class BufferizationDialect;
} // namespace bufferization
} // namespace mlir

namespace mlir {
namespace tensor {
class TensorDialect;
} // namespace tensor
} // namespace mlir

namespace mlir {
namespace LLVM {
class LLVMDialect;
} // namespace LLVM
} // namespace mlir

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
