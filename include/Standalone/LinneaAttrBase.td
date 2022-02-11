//===- LinneaAttrBase --------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_ATTRDEFS
#define LINNEA_ATTRDEFS

include "Standalone/LinneaDialect.td"

// All of the Linnea attributes will extend this class.
class Linnea_Attr<string name,
                      list<Trait> traits = []>
        : AttrDef<Linnea_Dialect, name, traits>;

// Linnea encoding attribute.
def LinneaMatrixEncodingAttr : Linnea_Attr<"LinneaMatrixEncoding"> {
  let mnemonic = "property";
  let description = [{ Attribute to describe propery of generic tensors.}];
  let parameters = (
    ins
    ArrayRefParameter<"LinneaMatrixEncodingAttr::MatrixProperty",
                      "type of matrix encoding"
                      >: $encoding
  );
  let genVerifyDecl = 1;
  
  let parser = [{
    if ($_parser.parseLess())
      return {};
    ArrayAttr properties;
    if ($_parser.parseAttribute(properties))
      return {};
    if ($_parser.parseGreater())
      return {};
    SmallVector<LinneaMatrixEncodingAttr::MatrixProperty, 4> mt;

    for (size_t i = 0, e = properties.size(); i < e; i++) {
      auto strAttr = properties[i].dyn_cast<StringAttr>();
      if (!strAttr) {
        $_parser.emitError($_parser.getNameLoc(),
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
        $_parser.emitError($_parser.getNameLoc(),
                           "unexpected matrix type: ") << strVal;
        return {};
      }
    }
    
    return LinneaMatrixEncodingAttr::getChecked([&$_parser] {
      return $_parser.emitError($_parser.getCurrentLocation()); },
      $_ctxt, mt);
  }];

  let extraClassDeclaration = [{
    enum class MatrixProperty {
      General,
      FullRank,
      Factored,
      Diagonal,
      UnitDiagonal,
      LowerTriangular,
      UpperTriangular,
      Symmetric,
      Square,
      SPD,
      SPSD
    };
  }];
}


#endif // LINNEA_ATTRDEFS
