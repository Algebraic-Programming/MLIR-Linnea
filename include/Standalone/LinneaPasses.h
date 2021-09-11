#ifndef LINNEA_PASSES_H
#define LINNEA_PASSES_H

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// namespace linnea {
std::unique_ptr<OperationPass<FuncOp>> createConvertLinneaToLinalgPass();
//} // end linnea
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "Standalone/LinneaPasses.h.inc"

#endif
