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
include "mlir/IR/AttrTypeBase.td"

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
  
  let hasCustomAssemblyFormat = 1; 

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
