//===- LinneaOps.td - Linnea dialect ops -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_OPS
#define LINNEA_OPS

include "Standalone/LinneaDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Standalone/LinneaTypeBase.td" 

def Linnea_PrintOp : Linnea_Op<"print"> {
  let summary = "print operation (for testing and debugging)";
  let arguments = (ins AnyLinneaType:$source);
  let assemblyFormat = "$source attr-dict `:` type($source)";
}

def Linnea_AllocOp : Linnea_Op<"alloc"> {
  let summary = "Materializes an unitialized linnea tensor";
  let arguments = (ins Variadic<Index>:$sizes);
  let results = (outs LinneaSymbolicMatrix:$result);
  let assemblyFormat = "`[` $sizes `]` attr-dict `:` type($result)";
}

def Linnea_DeallocOp : Linnea_Op<"dealloc"> {
  let summary = "Clean-up allocation";
  let arguments = (ins AnyTypeOf<[AnyRankedOrUnrankedMemRef,
                                  AnyTensor, 
                                  LinneaSymbolicMatrix]>:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Linnea_CastToBuiltinTensorOp : Linnea_Op<"cast_to_builtin_tensor", 
                                            [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Convert a `RankedTensorType` with a linnea attribute to a builtin `tensor`";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `->` qualified(type($output))
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Linnea_CastFromBuiltinTensorOp : Linnea_Op<"cast_from_builtin_tensor", 
                                              [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Convert a `RankedTensorType` to a `RankedTensorType` with linnea attribute";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `->` qualified(type($output))
  }];
}

def Linnea_ToBuiltinTensorOp : Linnea_Op<"to_builtin_tensor", 
                                        [NoSideEffect]> {
  let summary = "Convert a `LinneaSymbolicMatrix` to a builtin `tensor`";
  let arguments = (ins AnyLinneaType:$operand);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];
}

def Linnea_FromBuiltinTensorOp : Linnea_Op<"from_builtin_tensor", 
                                          [NoSideEffect]> {
  let summary = "Convert a `tensor` to a `LinneaSymbolicMatrix`";
  let arguments = (ins AnyTensor:$operand);
  let results = (outs LinneaSymbolicMatrix:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];
}

def Linnea_FillOp : Linnea_Op<"fill", [NoSideEffect]> {
  let summary = "The `linnea.fillOp` fill a linnea container with a given value";
  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnySignlessInteger]>:$value,
    AnyLinneaType:$output);
  let results = (outs Optional<AnyLinneaType>:$result);

  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$output)>
  ];
  
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Linnea_EquationOp : Linnea_Op<"equation", 
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Operation wrapper to contains a set of linnea operations.";
  let description = [{
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
    Region &getRegion() { return getOperation()->getRegion(0); }
    Block &getBody() { return getOperation()->getRegion(0).front(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Linnea_YieldOp : Linnea_Op<"yield", 
    [Terminator, HasParent<"EquationOp">]> {
  let summary = "The `linnea.yield` returns value from `linnea.equation`.";
  let arguments = (ins AnyLinneaType:$input);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def Linnea_InverseOpLow : Linnea_Op<"inverse.low", [NoSideEffect]> {
    let summary = "Compute inverse";
    let description = [{
      It is a low-level linnea operation (after property
      propagation and can live outisde a `linnea.equation`.
    }];

    let arguments = (ins AnyLinneaType:$input);
    let results = (outs AnyLinneaType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def Linnea_InverseOpHigh : Linnea_Op<"inverse.high", 
                                    [HasParent<"EquationOp">, NoSideEffect]> {
    let summary = "Compute inverse";
    let description = [{
      It is an high-level linnea operation. It live
      only within a `linnea.equation`.
    }];

    let arguments = (ins AnyLinneaType:$input);
    let results = (outs AnyLinneaType:$output);

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;
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

    let hasVerifier = 1;
}

def Linnea_MulOpHigh : Linnea_Op<"mul.high", 
                                [HasParent<"EquationOp">, NoSideEffect]> {
    let summary = "Variadic matrix multiplication";
    let description = [{
        The `linnea.mul.high` represents a variadic multiplication.
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

    let hasVerifier = 1;
}

def Linnea_MulOpLow : Linnea_Op<"mul.low", [NoSideEffect]> {
    let summary = "Matrix multiplication";
    let description = [{
        The `linnea.mul.low` represents a 2d multiplication.
        It is an low-level linnea operation ready to be converted
        1:1 with linalg.generic.

        Example:

        ```mlir
          %2 = linnea.mul %0, %1 : { semirings = "min-plus" } (...)
        ```
    }];

    let arguments = (ins AnyLinneaType:$left, 
                         AnyLinneaType:$right, 
                         StrAttr:$semirings);
    let results = (outs AnyLinneaType:$output);

    let assemblyFormat = [{
      $left `,` $right attr-dict `:` type($left) `,` type($right) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "ValueRange":$input)>
    ];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def Linnea_AddOpHigh : Linnea_Op<"add.high", 
                                [HasParent<"EquationOp">, NoSideEffect]> {
    let summary = "Variadic matrix addition";
    let description = [{
        The `linnea.add.high` represents a variadic addition.
        It is an high-level linnea operation which must be nested within
        a linnea.equation operation.

        Example:

        ```mlir
        { // linnea.equation
          %2 = linnea.add %0, %1, %2 : (...)
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

    let hasVerifier = 1;
}

def Linnea_AddOpLow : Linnea_Op<"add.low", [NoSideEffect]> {
    let summary = "Matrix addition";
    let description = [{
        The `linnea.add.low` represents a 2d addition.
        It is an low-level linnea operation ready to be converted
        1:1 with linalg.generic.

        Example:

        ```mlir
          %2 = linnea.add %0, %1 : { semirings = "min-plus" } (...)
        ```
    }];

    let arguments = (ins AnyLinneaType:$left, 
                         AnyLinneaType:$right,
                         StrAttr:$semirings);
    let results = (outs AnyLinneaType:$output);

    let assemblyFormat = [{
      $left `,` $right attr-dict `:` type($left) `,` type($right) `->` type($output)
    }];

    let builders = [
      OpBuilder<(ins "ValueRange":$input)>
    ];

    let hasVerifier = 1;
}

#endif // LINNEA_OPS
