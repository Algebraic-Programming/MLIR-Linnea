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
    ArrayRef<LinneaMatrixEncodingAttr::MatrixProperty>) {
  return success();
}

LinneaMatrixEncodingAttr getMatrixEncodingAttr(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<LinneaMatrixEncodingAttr>();
  return nullptr;
}

template <LinneaMatrixEncodingAttr::MatrixProperty T>
bool isT(Type type) {
  auto encoding = getMatrixEncodingAttr(type);
  if (!encoding)
    return false;
  if (llvm::is_contained(encoding.getEncodingType(), T))
    return true;
  return false;
}

bool hasSPDAttr(Type type) {
  return isT<LinneaMatrixEncodingAttr::MatrixProperty::SPD>(type);
}

bool hasLowerTriangularAttr(Type type) {
  return isT<LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular>(type);
}

bool hasUpperTriangularAttr(Type type) {
  return isT<LinneaMatrixEncodingAttr::MatrixProperty::UpperTriangular>(type);
}

void LinneaMatrixEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<[";

  for (size_t i = 0, e = getEncodingType().size(); i < e; i++) {
    switch (getEncodingType()[i]) {
    case MatrixProperty::General:
      printer << "\"general\"";
      break;
    case MatrixProperty::FullRank:
      printer << "\"fullrank\"";
      break;
    case MatrixProperty::Factored:
      printer << "\"factored\"";
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
    case MatrixProperty::Square:
      printer << "\"square\"";
      break;
    case MatrixProperty::Identity:
      printer << "\"identity\"";
      break;
    }
    if (i != e - 1)
      printer << ", ";
  }
  printer << "]>";
}
