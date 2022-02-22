//===-- Standalone-c/LinneaDialect.h - C API Dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_LINNEA_H
#define MLIR_C_DIALECT_LINNEA_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Linnea, linnea);

/// Attributes encoding.
enum MlirLinneaMatrixEncoding {
  MLIR_LINNEA_MATRIX_PROPERTY_GENERAL,
  MLIR_LINNEA_MATRIX_PROPERTY_FULLRANK,
  MLIR_LINNEA_MATRIX_PROPERTY_DIAGONAL,
  MLIR_LINNEA_MATRIX_PROPERTY_UNITDIAGONAL,
  MLIR_LINNEA_MATRIX_PROPERTY_LOWERTRIANGULAR,
  MLIR_LINNEA_MATRIX_PROPERTY_UPPERTRIANGULAR,
  MLIR_LINNEA_MATRIX_PROPERTY_SYMMETRIC,
  MLIR_LINNEA_MATRIX_PROPERTY_SPD,
  MLIR_LINNEA_MATRIX_PROPERTY_SPSD,
  MLIR_LINNEA_MATRIX_PROPERTY_SQUARE,
  MLIR_LINNEA_MATRIX_PROPERTY_FACTORED,
};

//===---------------------------------------------------------------------===//
// Custom filler for equationOp
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void mlirLinneaEquationOpFillRegion(MlirOperation op);

//===---------------------------------------------------------------------===//
// LinneaMatrixEncodingAttr
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool
mlirAttributeIsLinneaMatrixEncodingAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirLinneaAttributeMatrixEncodingAttrGet(
    MlirContext ctx, intptr_t numProperties,
    enum MlirLinneaMatrixEncoding const *properties);

//===---------------------------------------------------------------------===//
// MatrixType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsLinneaMatrixType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirLinneaMatrixTypeGet(MlirContext ctx,
                                                    MlirAttribute attr,
                                                    intptr_t rank,
                                                    const int64_t *shape,
                                                    MlirType elementType);

//===---------------------------------------------------------------------===//
// TermType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsLinneaTermType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirLinneaTermTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#include "Standalone/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_LINNEA_H
