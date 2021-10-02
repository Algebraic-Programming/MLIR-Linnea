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
include "LinneaTypeBase.td" 

def Linnea_InverseOp : Linnea_Op<"inverse", [NoSideEffect]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins LinneaSymbolicMatrix:$input);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let verifier = [{ return ::verifyInverseOp(*this); }];

    let hasCanonicalizer = 1;
}

def Linnea_TransposeOp : Linnea_Op<"transpose", [NoSideEffect]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins LinneaSymbolicMatrix:$input);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let verifier = [{ return ::verifyTransposeOp(*this); }];
}

def Linnea_CholeskyOp : Linnea_Op<"cholesky", [NoSideEffect]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins LinneaSymbolicMatrix:$input);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let verifier = [{ return ::verifyCholeskyOp(*this); }];
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

    let arguments = (ins Variadic<LinneaSymbolicMatrix>:$input);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let hasCanonicalizer = 1;

    let builders = [
      OpBuilder<(ins "ValueRange":$input)>
    ];

    let verifier = [{ return ::verifyMulOp(*this); }];
}

#endif // LINNEA_OPS
