//===- LinneaCompilerPipeline.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaPasses.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linnea;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

#define DEBUG_TYPE "linnea-compiler-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct LinneaCompilerPipeline
    : public LinneaCompilerBase<LinneaCompilerPipeline> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                tensor::TensorDialect, linalg::LinalgDialect, scf::SCFDialect,
                arith::ArithmeticDialect, LLVM::LLVMDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
  }
};

void LinneaCompilerPipeline::runOnOperation() {
  OpPassManager pm("builtin.module");
  // optimize and propagate properties.
  pm.addPass(createLinneaPropertyPropagationPass());
  // type conversion at the function boundaries.
  pm.addPass(createLinneaFuncTypeConversion());
  // lower to linalg.
  pm.addNestedPass<func::FuncOp>(createConvertLinneaToLinalgPass());
  // lowert to loops.
  pm.addPass(createConvertLinneaToLoopsPass());
  // finalize type conversion. From this point on
  // we have builtin types.
  pm.addPass(createLinneaFinalizeFuncTypeConversion());
  // --canonicalize
  pm.addPass(createCanonicalizerPass());
  //  --linalg-bufferize
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  // --func-bufferize
  pm.addPass(mlir::func::createFuncBufferizePass());
  // --arith-bufferize
  pm.addPass(arith::createConstantBufferizePass());
  // --tensor-bufferize
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  // --finalizing-bufferize
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  // Remove extra copy operations introduced by bufferization.
  // We will remove this pass once bufferization is fixed.
  // --remove-extra-copy-operations.
  pm.addPass(createLinneaCopyRemoval());
  // --convert-linalg-to-loops
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  // --convert-vector-to-scf
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  // --convert-scf-to-cf
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  // --convert-memref-to-llvm
  pm.addPass(createMemRefToLLVMPass());
  // --convert-artih-to-llvm
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // --convert-vector-to-llvm
  pm.addPass(createConvertVectorToLLVMPass());
  // --convert-std-to-llvm
  pm.addPass(createConvertFuncToLLVMPass());
  // --reconcile-unrealized-casts
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(runPipeline(pm, getOperation())))
    signalPassFailure();

  return;
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createLinneaCompilerPipeline() {
  return std::make_unique<LinneaCompilerPipeline>();
}
