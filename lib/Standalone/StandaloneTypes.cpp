#include "Standalone/StandaloneTypes.h"
#include "Standalone/StandaloneDialect.h"

using namespace mlir;
using namespace mlir::standalone;

LogicalResult MatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TensorType tensor, Attribute attribute) {
  if (tensor.getRank() != 2)
    return emitError() << "expect a 2-d tensor for 'matrixType'";
  if (!tensor.isa<RankedTensorType>())
    return emitError() << "non ranked-tensor type passed to 'matrixType'";
  return success();
}

void MatrixType::print(DialectAsmPrinter &printer) const {
  printer << MatrixType::getMnemonic();
  printer << "<";
  printer.printType(getParam());
  printer << ",";
  printer.printAttribute(getEncoding());
  printer << ">";
}
