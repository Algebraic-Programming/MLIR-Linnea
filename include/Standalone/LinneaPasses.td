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
  let constructor = "mlir::linnea::createConvertLinneaToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
}

def LinneaConvertToLoops : Pass<"convert-linnea-to-loops", "ModuleOp"> {
  let summary = "Lower linnea operations to loop";
  let constructor = "mlir::linnea::createConvertLinneaToLoopsPass()";
  let dependentDialects = ["scf::SCFDialect", 
                           "memref::MemRefDialect", 
                           "bufferization::BufferizationDialect",
                           "arith::ArithmeticDialect",
                           "tensor::TensorDialect",
                           "vector::VectorDialect"];
}

def LinneaComprehensivePropertyPropagation :
    Pass<"comprehensive-properties-propagation", "ModuleOp"> {
  let summary = "property propagation";
  let description = [{ todo }];
  let constructor = "mlir::linnea::createLinneaComprehensivePropertyPropagationPass()";
}

def LinneaFuncTypeConversion : Pass<"linnea-func-type-conversion", "ModuleOp"> {
  let summary = "Conversion to builtin tensors with linnea attributes from linnea matrix type";
  let constructor = "mlir::linnea::createLinneaFuncTypeConversion()";
}

def LinneaFinalizeFuncTypeConversion : Pass<"linnea-finalize-func-type-conversion", "ModuleOp"> {
  let summary = "Conversion to builtin tensors with linnea attributes to builtin tensor";
  let constructor = "mlir::linnea::createLinneaFinalizeFuncTypeConversion()";
}

#endif
