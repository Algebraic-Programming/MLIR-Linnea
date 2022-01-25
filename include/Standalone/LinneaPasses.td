//===- LinneaPasses.td -------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINNEA_PASSES
#define MLIR_DIALECT_LINNEA_PASSES

include "mlir/Pass/PassBase.td"

def LinneaConvertToLinalg : FunctionPass<"convert-linnea-to-linalg"> {
  let summary = "Lower linnea operations to linalg";
  let constructor = "mlir::createConvertLinneaToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
}

def LinneaConvertToLoops : FunctionPass<"convert-linnea-to-loops"> {
  let summary = "Lower linnea operations to loop";
  let constructor = "mlir::createConvertLinneaToLoopsPass()";
  let dependentDialects = ["scf::SCFDialect", "memref::MemRefDialect"];
}

def LinneaComprehensivePropertyPropagation :
    Pass<"comprehensive-properties-propagation", "ModuleOp"> {
  let summary = "property propagation";
  let description = [{ todo }];
  let constructor = "mlir::createLinneaComprehensivePropertyPropagationPass()";
}

#endif
