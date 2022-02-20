//===- standalone-cap-demo.c - Simple demo of C-API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: standalone-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "Standalone-c/LinneaDialect.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();

  // CHECK-LABEL: testDialectRegistration
  fprintf(stderr, "testDialectRegistration\n");
  mlirRegisterAllDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__linnea__(), ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString("%0 = arith.constant 2 : i32\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 2 : i32
  mlirOperationDump(op);

  // CHECK-LABEL: testMatrixTypeAttr
  fprintf(stderr, "testMatrixTypeAttr\n");
  const char *originalAsmAttr = "#linnea.property<[\"general\"]>";
  MlirAttribute originalAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString(originalAsmAttr));
  // CHECK: isa: 1
  fprintf(stderr, "isa: %d\n",
          mlirAttributeIsLinneaMatrixEncodingAttr(originalAttr));
  const enum MlirLinneaMatrixEncoding property =
      MLIR_LINNEA_MATRIX_PROPERTY_GENERAL;
  MlirAttribute propertyMatrix =
      mlirLinneaAttributeMatrixEncodingAttrGet(ctx, 1, &property);
  // CHECK: #linnea.property<["general"]>
  mlirAttributeDump(propertyMatrix);

  // CHECK-LABEL: testMatrixType
  fprintf(stderr, "testMatrixType\n");
  const char *originalAsmMatrixType =
      "!linnea.matrix<#linnea.property<[\"general\"]>, [32, 32], f32>";
  MlirType originalMatrixType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString(originalAsmMatrixType));
  // CHECK: isa: 1
  fprintf(stderr, "isa: %d\n", mlirTypeIsLinneaMatrixType(originalMatrixType));
  const int64_t sizes[2] = {23, 23};
  MlirType matrix = mlirLinneaMatrixTypeGet(ctx, propertyMatrix, 2, sizes,
                                            mlirF32TypeGet(ctx));
  // CHECK: !linnea.matrix<#linnea.property<["general"]>, [23, 23], f32>
  mlirTypeDump(matrix);

  // CHECK-LABEL: testTermType
  fprintf(stderr, "testTermType\n");
  const char *originalAsmTermType = "!linnea.term";
  MlirType originalTermType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString(originalAsmTermType));
  // CHECK: isa: 1
  fprintf(stderr, "isa: %d\n", mlirTypeIsLinneaTermType(originalTermType));
  MlirType term = mlirLinneaTermTypeGet(ctx);
  // CHECK: !linnea.term
  mlirTypeDump(term);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
