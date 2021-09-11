#ifndef MLIR_DIALECT_LINNEA_PASSES
#define MLIR_DIALECT_LINNEA_PASSES

include "mlir/Pass/PassBase.td"

def LinneaLowerToLinalg : FunctionPass<"convert-linnea-to-linalg"> {
  let summary = "Lower linnea operations to linalg";
  let constructor = "mlir::createConvertLinneaToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

#endif
