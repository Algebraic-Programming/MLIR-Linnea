//===- LinneaTypes.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaTypes.h"
#include "Standalone/LinneaDialect.h"

using namespace mlir;
using namespace mlir::linnea;

LogicalResult MatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<MatrixType::MatrixProperty> property,
                                 ArrayRef<int64_t> dims, Type elementType) {
  return success();
}

template <MatrixType::MatrixProperty T>
bool isT(MatrixType type) {
  assert(type && "must be valid");
  ArrayRef<MatrixType::MatrixProperty> properties = type.getProperty();
  if (llvm::is_contained(properties, T))
    return true;
  return false;
}

bool isSPD(MatrixType type) {
  return isT<MatrixType::MatrixProperty::SPD>(type);
}

bool isLowerTriangular(MatrixType type) {
  return isT<MatrixType::MatrixProperty::LowerTriangular>(type);
}

bool isUpperTriangular(MatrixType type) {
  return isT<MatrixType::MatrixProperty::UpperTriangular>(type);
}

void MatrixType::print(AsmPrinter &printer) const {
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
    case MatrixProperty::Square:
      printer << "\"square\"";
      break;
    case MatrixProperty::Factored:
      printer << "\"factored\"";
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
  printer << ", ";
  printer << getElementType();
  printer << ">";
}
