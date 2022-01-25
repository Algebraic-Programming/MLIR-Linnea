//===- LinneaOps.td - Linnea dialect ops -------------------*- tablegen -*-===//
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

def Linnea_FillOp : Linnea_Op<"fill"> {
  let summary = "The `linnea.fillOp` fill a linnea container with a given value";
  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnySignlessInteger]>:$value,
    AnyLinneaType:$output);

  let assemblyFormat = [{
    `(` $value `,` $output `)` attr-dict `:` 
        type($value) `,` type($output)
  }];
  
  let verifier = [{ return ::verifyFillOp(*this); }];
}

def Linnea_EquationOp : Linnea_Op<"equation", 
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Operation wrapper to contains a set of linnea operations.";
  let description = [{
    ```mlir
    ```
    `linnea.equation` is introduced as a wrapper for `linnea` dialect
    operations which makes analysis and transformation simpler.
    The `linnea.equationOp` operation has a single region with a single
    basic block attached. The block is terminated by `linnea.yieldOp` 
    which returns the result.
    The `linnea.equation` region allows implicit capture.
  }];
  
  let results = (outs LinneaTerm:$output);

  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    Region &getBody() { return getOperation()->getRegion(0); }
  }];

  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
}

def Linnea_YieldOp : Linnea_Op<"yield", 
    [Terminator, HasParent<"EquationOp">]> {
  let summary = "The `linnea.yield` returns value from `linnea.equation`.";
  let arguments = (ins AnyLinneaType:$input);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def Linnea_InverseOp : Linnea_Op<"inverse", [HasParent<"EquationOp">]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins AnyLinneaType:$input);
    let results = (outs LinneaTerm:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let verifier = [{ return ::verifyInverseOp(*this); }];

    let hasCanonicalizer = 1;
}

def Linnea_TransposeOp : Linnea_Op<"transpose", [HasParent<"EquationOp">]> {
    let summary = "Add doc.";
    let description = [{
    
    }];

    let arguments = (ins AnyLinneaType:$input);
    let results = (outs LinneaTerm:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let verifier = [{ return ::verifyTransposeOp(*this); }];
}

def Linnea_MulOpHigh : Linnea_Op<"mul.high", [HasParent<"EquationOp">]> {
    let summary = "Variadic matrix multiplication";
    let description = [{
        The `linnea.mul` represents a variadic multiplication.
        It is an high-level linnea operation which must be nested within
        a linnea.equation operation.

        Example:

        ```mlir
        { // linnea.equation
          %2 = linnea.mul %0, %1, %2 : (...)
        } // end linnea.equation
        ```
    }];

    let arguments = (ins Variadic<AnyLinneaType>:$input);
    let results = (outs AnyLinneaType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "ValueRange":$input)>
    ];

    let verifier = [{ return ::verifyMulOp(*this); }];
}

// def MulOLowOperand : 2DTensorOf<[AnyLinneaType]>;

def Linnea_MulOpLow : Linnea_Op<"mul.low"> {
    let summary = "Matrix multiplication";
    let description = [{
        The `linnea.mul` represents a 2d multiplication.
        It is an low-level linnea operation ready to be converted
        1:1 with linalg.generic.

        Example:

        ```mlir
          %2 = linnea.mul %0, %1 : (...)
        ```
    }];

    let arguments = (ins AnyLinneaType:$a, AnyLinneaType:$b);
    let results = (outs AnyLinneaType:$output);

    let assemblyFormat = [{
      $a `,` $b attr-dict `:` type($a) `,` type($b) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "ValueRange":$input)>
    ];

    let verifier = [{ return ::verifyMulOp(*this); }];
}

#endif // LINNEA_OPS
