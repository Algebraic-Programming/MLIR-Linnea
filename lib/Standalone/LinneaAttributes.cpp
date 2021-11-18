//===- LinneaAttributes.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaDialect.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linnea;

LogicalResult LinneaMatrixEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<LinneaMatrixEncodingAttr::MatrixType>) {
  return success();
}

LinneaMatrixEncodingAttr getMatrixEncodingAttr(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<LinneaMatrixEncodingAttr>();
  return nullptr;
}

template <LinneaMatrixEncodingAttr::MatrixType T>
bool isT(Type type) {
  auto encoding = getMatrixEncodingAttr(type);
  if (!encoding)
    return false;
  if (llvm::is_contained(encoding.getEncodingType(), T))
    return true;
  return false;
}

bool hasSPDAttr(Type type) {
  return isT<LinneaMatrixEncodingAttr::MatrixType::SPD>(type);
}

bool hasLowerTriangularAttr(Type type) {
  return isT<LinneaMatrixEncodingAttr::MatrixType::LowerTriangular>(type);
}

bool hasUpperTriangularAttr(Type type) {
  return isT<LinneaMatrixEncodingAttr::MatrixType::UpperTriangular>(type);
}

void LinneaMatrixEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{p = [";

  for (size_t i = 0, e = getEncodingType().size(); i < e; i++) {
    switch (getEncodingType()[i]) {
    case MatrixType::General:
      printer << "\"general\"";
      break;
    case MatrixType::FullRank:
      printer << "\"fullrank\"";
      break;
    case MatrixType::Factored:
      printer << "\"factored\"";
      break;
    case MatrixType::Diagonal:
      printer << "\"diagonal\"";
      break;
    case MatrixType::UnitDiagonal:
      printer << "\"unitdiagonal\"";
      break;
    case MatrixType::LowerTriangular:
      printer << "\"lowerTri\"";
      break;
    case MatrixType::UpperTriangular:
      printer << "\"upperTri\"";
      break;
    case MatrixType::Symmetric:
      printer << "\"symm\"";
      break;
    case MatrixType::SPD:
      printer << "\"spd\"";
      break;
    case MatrixType::SPSD:
      printer << "\"spsd\"";
      break;
    case MatrixType::Square:
      printer << "\"square\"";
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
