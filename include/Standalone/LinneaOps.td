//===- LinneaOps.td - Linnea dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_OPS
#define LINNEA_OPS

include "LinneaDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Linnea_Matrix : 2DTensorOf<[AnyType]>;

def Linnea_InverseOp : Linnea_Op<"inverse", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins Linnea_Matrix:$input);
    let results = (outs Linnea_Matrix:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input)
    }];

    let hasCanonicalizeMethod = 1;
}

def Linnea_TransOp : Linnea_Op<"transpose", [NoSideEffect, 
                                             SameOperandsAndResultType]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins Linnea_Matrix:$input);
    let results = (outs Linnea_Matrix:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input)
    }];
}

def Linnea_CholeskyOp : Linnea_Op<"cholesky", [NoSideEffect, 
                                               SameOperandsAndResultType]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins Linnea_Matrix:$input);
    let results = (outs Linnea_Matrix:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input)
    }];
}

def Linnea_MulOp : Linnea_Op<"mul", [NoSideEffect]> {
    let summary = "Add doc.";
    let description = [{
        The `standalone.inverse` (add doc)

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = linnea.inverse %0 : i32
        ```
    }];

    let arguments = (ins Variadic<Linnea_Matrix>:$input);
    let results = (outs Linnea_Matrix:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];
}

#endif // LINNEA_OPS
