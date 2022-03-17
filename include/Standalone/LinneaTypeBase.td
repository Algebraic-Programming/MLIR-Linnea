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
include "mlir/IR/BuiltinTypes.td"
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

// TODO: Make this more modular. In some way LinneaSymbolicMatrixIdentity
// should derive from LinneaSymbolicMatrix.  See shapedType in MLIR.

def LinneaSymbolicMatrixIdentity : Linnea_Type<"Identity"> {
  let mnemonic = "identity";
  
  let summary = "An identity matrix type.";
  let description = [{
    The identity matrix.
  }];
  let genVerifyDecl = 1;
  
  let parameters = (ins
    ArrayRefParameter<"int64_t", "dimensions" 
                     >:$dims,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
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

  let hasCustomAssemblyFormat = 1; 
}

def AnyLinneaType : AnyTypeOf<[LinneaSymbolicMatrix, 
                               LinneaTerm,
                               LinneaSymbolicMatrixIdentity]>;

#endif // LINNEA_TYPE_BASE 
