#include "Standalone/StandaloneAttributes.h"
#include "Standalone/StandaloneDialect.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::standalone;

LogicalResult StandaloneMatrixEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<StandaloneMatrixEncodingAttr::MatrixType>) {
  return success();
}

void StandaloneMatrixEncodingAttr::print(DialectAsmPrinter &printer) const {
  printer << "matrix.encoding";
}
