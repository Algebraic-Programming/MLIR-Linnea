#include "Standalone/StandaloneTypes.h"
#include "Standalone/StandaloneDialect.h"

using namespace mlir;
using namespace mlir::standalone;

LogicalResult MatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TensorType tensor, Attribute attribute) {
  if (!tensor.isa<RankedTensorType>())
    return emitError() << "non ranked-tensor type passed to 'matrixType'";
  return success();
}
