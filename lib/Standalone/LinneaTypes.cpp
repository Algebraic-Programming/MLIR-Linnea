//===- LinneaTypes.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaTypes.h"
#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaDialect.h"

using namespace mlir;
using namespace mlir::linnea;

//===----------------------------------------------------------------------===//
// MatrixType
//===----------------------------------------------------------------------===//

LogicalResult MatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 LinneaMatrixEncodingAttr property,
                                 ArrayRef<int64_t> dims, Type elementType) {
  // a matrix must be 2-dimensional.
  if (dims.size() != 2)
    return failure();
  return success();
}

void MatrixType::print(AsmPrinter &printer) const {
  printer << "<";
  printer << getProperty();
  printer << ", [";
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

//===----------------------------------------------------------------------===//
// IdentityType
//===----------------------------------------------------------------------===//

LogicalResult IdentityType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   ArrayRef<int64_t> dims, Type elementType) {
  if (dims.size() != 2)
    return failure();
  return success();
}

void IdentityType::print(AsmPrinter &printer) const {
  printer << "<[";
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
