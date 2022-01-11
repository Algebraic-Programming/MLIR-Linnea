//===- LinneaTypeBase --------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_TYPE_BASE
#define LINNEA_TYPE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "Standalone/LinneaDialect.td"

class Linnea_Type<string name>
  : TypeDef<Linnea_Dialect, name>;

def LinneaTerm : Linnea_Type<"Term"> {
  let mnemonic = "term";
  let summary = "A symbolic tensor";
  let description = [{
    A generic type similar to anyType.
  }];
}

def LinneaSymbolicMatrix : Linnea_Type<"Matrix"> {
  let mnemonic = "matrix";

  let summary = "A symbolic matrix type.";
  let description = [{
    A 2-dimensional tensor.
  }];
  let genVerifyDecl = 1;

  let parameters = (ins
    "LinneaMatrixEncodingAttr":$property,
    ArrayRefParameter<"int64_t", "dimensions"
                     >:$dims,
    "Type":$elementType
  );

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    LinneaMatrixEncodingAttr properties;
    if ($_parser.parseAttribute(properties))
      return Type();
    if ($_parser.parseComma())
      return Type();
    ArrayAttr dimensions;
    if ($_parser.parseAttribute(dimensions))
      return Type();
    
    SmallVector<int64_t, 4> dims; 
    for (size_t i = 0, e = dimensions.size(); i < e; i++) {
      auto intAttr = dimensions[i].dyn_cast<IntegerAttr>();
      if (!intAttr) {
        $_parser.emitError($_parser.getNameLoc(),
                           "expect int attribute for matrix type");
        return Type();
      }
      dims.push_back(intAttr.getInt());
    }

    if ($_parser.parseComma())
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();

    if ($_parser.parseGreater())
      return Type();
    return MatrixType::getChecked([&$_parser] { 
      return $_parser.emitError($_parser.getCurrentLocation()); }, 
             $_ctxt, properties, dims, elementType);
  }]; 
}

def AnyLinneaType : AnyTypeOf<[LinneaSymbolicMatrix, LinneaTerm]>;

#endif // LINNEA_TYPE_BASE 
