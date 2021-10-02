#include "Standalone/LinneaTypes.h"
#include "Standalone/LinneaDialect.h"

using namespace mlir;
using namespace mlir::linnea;

LogicalResult
MatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                   llvm::ArrayRef<MatrixType::MatrixProperty> property,
                   llvm::ArrayRef<int64_t> dims) {
  return success();
}

void MatrixType::print(DialectAsmPrinter &printer) const {
  printer << MatrixType::getMnemonic();
  printer << "<[";

  for (size_t i = 0, e = getProperty().size(); i < e; i++) {
    switch (getProperty()[i]) {
    case MatrixProperty::General:
      printer << "\"general\"";
      break;
    case MatrixProperty::FullRank:
      printer << "\"fullrank\"";
      break;
    case MatrixProperty::Diagonal:
      printer << "\"diagonal\"";
      break;
    case MatrixProperty::UnitDiagonal:
      printer << "\"unitdiagonal\"";
      break;
    case MatrixProperty::LowerTriangular:
      printer << "\"lowerTri\"";
      break;
    case MatrixProperty::UpperTriangular:
      printer << "\"upperTri\"";
      break;
    case MatrixProperty::Symmetric:
      printer << "\"symm\"";
      break;
    case MatrixProperty::SPD:
      printer << "\"spd\"";
      break;
    case MatrixProperty::SPSD:
      printer << "\"spsd\"";
      break;
    case MatrixProperty::Identity:
      printer << "\"identity\"";
      break;
    }
    if (i != e - 1)
      printer << ", ";
  }
  printer << "], ";
  printer << "[";
  for (size_t i = 0, e = getDims().size(); i < e; i++) {
    printer << getDims()[i];
    if (i != e - 1)
      printer << ", ";
  }
  printer << "]";
  printer << ">";
}
