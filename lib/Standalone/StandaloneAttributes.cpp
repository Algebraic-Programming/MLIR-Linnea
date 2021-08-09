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
  printer << "matrix_encoding<{encodingType = [";

  for (size_t i = 0, e = getEncodingType().size(); i < e; i++) {
    switch (getEncodingType()[i]) {
    case MatrixType::Diagonal:
      printer << "diagonal";
      break;
    case MatrixType::UnitDiagonal:
      printer << "unitdiagonal";
      break;
    case MatrixType::LowerTriangular:
      printer << "lowertriangular";
      break;
    case MatrixType::UpperTriangular:
      printer << "uppertriangular";
      break;
    case MatrixType::Symmetric:
      printer << "symmetric";
      break;
    case MatrixType::SPD:
      printer << "spd";
      break;
    case MatrixType::SPSD:
      printer << "spsd";
      break;
    case MatrixType::Identity:
      printer << "\"identity\"";
      break;
    }
    if (i != e - 1)
      printer << ", ";
  }
  printer << "]}>";
}
