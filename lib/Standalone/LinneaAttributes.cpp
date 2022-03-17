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

Attribute LinneaMatrixEncodingAttr::parse(AsmParser &parser, Type t) {
  if (failed(parser.parseLess()))
    return {};
  ArrayAttr properties;
  if (failed(parser.parseAttribute(properties)))
    return {};
  if (failed(parser.parseGreater()))
    return {};
  SmallVector<LinneaMatrixEncodingAttr::MatrixProperty, 4> mt;

  for (size_t i = 0, e = properties.size(); i < e; i++) {
    auto strAttr = properties[i].dyn_cast<StringAttr>();
    if (!strAttr) {
      parser.emitError(parser.getNameLoc(),
                       "expect string attribute for matrix type");
      return {};
    }
    auto strVal = strAttr.getValue();
    if (strVal == "general")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::General);
    else if (strVal == "fullrank")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::FullRank);
    else if (strVal == "diagonal")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Diagonal);
    else if (strVal == "unitdiagonal")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::UnitDiagonal);
    else if (strVal == "lowerTri")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular);
    else if (strVal == "upperTri")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::UpperTriangular);
    else if (strVal == "symmetric")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Symmetric);
    else if (strVal == "spd")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::SPD);
    else if (strVal == "spds")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::SPSD);
    else if (strVal == "square")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Square);
    else if (strVal == "factored")
      mt.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Factored);
    else {
      parser.emitError(parser.getNameLoc(), "unexpected matrix type: ")
          << strVal;
      return {};
    }
  }

  return LinneaMatrixEncodingAttr::getChecked(
      [&parser] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), {mt});
}

void LinneaMatrixEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<[";

  for (size_t i = 0, e = getEncoding().size(); i < e; i++) {
    switch (getEncoding()[i]) {
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
    }
    if (i != e - 1)
      printer << ", ";
  }
  printer << "]>";
}
