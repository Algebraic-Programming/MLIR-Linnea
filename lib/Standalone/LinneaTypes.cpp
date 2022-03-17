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

Type MatrixType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();
  LinneaMatrixEncodingAttr properties;
  if (failed(parser.parseAttribute(properties)))
    return Type();
  if (failed(parser.parseComma()))
    return Type();
  ArrayAttr dimensions;
  if (failed(parser.parseAttribute(dimensions)))
    return Type();

  SmallVector<int64_t, 4> dims;
  for (size_t i = 0, e = dimensions.size(); i < e; i++) {
    auto intAttr = dimensions[i].dyn_cast<IntegerAttr>();
    if (!intAttr) {
      parser.emitError(parser.getNameLoc(),
                       "expect int attribute for matrix type");
      return Type();
    }
    dims.push_back(intAttr.getInt());
  }

  if (failed(parser.parseComma()))
    return Type();
  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();

  if (failed(parser.parseGreater()))
    return Type();
  return MatrixType::getChecked(
      [&parser] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), properties, {dims}, elementType);
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

Type IdentityType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();
  ArrayAttr dimensions;
  if (failed(parser.parseAttribute(dimensions)))
    return Type();
  SmallVector<int64_t, 2> dims;
  for (size_t i = 0, e = dimensions.size(); i < e; i++) {
    auto intAttr = dimensions[i].dyn_cast<IntegerAttr>();
    if (!intAttr) {
      parser.emitError(parser.getNameLoc(),
                       "expect int attribute for identity type");
      return Type();
    }
    dims.push_back(intAttr.getInt());
  }
  if (failed(parser.parseComma()))
    return Type();
  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();
  if (failed(parser.parseGreater()))
    return Type();

  MLIRContext *ctx = parser.getContext();
  return IdentityType::getChecked(
      [&] { return parser.emitError(parser.getNameLoc()); }, ctx, {dims},
      elementType);
}
