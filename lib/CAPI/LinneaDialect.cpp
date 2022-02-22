//===- LinalgDialect.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/LinneaDialect.h"
#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaOps.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::linnea;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linnea, linnea,
                                      mlir::linnea::LinneaDialect)

void mlirLinneaEquationOpFillRegion(MlirOperation mlirOp) {
  Operation *op = unwrap(mlirOp);
  auto equationOp = cast_or_null<EquationOp>(op);
  assert(equationOp);
  Region &region = op->getRegion(0);
  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Block *block = b.createBlock(&region);
  b.setInsertionPointToStart(block);
}

bool mlirAttributeIsLinneaMatrixEncodingAttr(MlirAttribute attr) {
  return unwrap(attr).isa<LinneaMatrixEncodingAttr>();
}

// really bad. fix it.
LinneaMatrixEncodingAttr::MatrixProperty
getCppProperty(MlirLinneaMatrixEncoding p) {
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_GENERAL)
    return LinneaMatrixEncodingAttr::MatrixProperty::General;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_FULLRANK)
    return LinneaMatrixEncodingAttr::MatrixProperty::FullRank;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_FACTORED)
    return LinneaMatrixEncodingAttr::MatrixProperty::Factored;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_DIAGONAL)
    return LinneaMatrixEncodingAttr::MatrixProperty::Diagonal;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_UNITDIAGONAL)
    return LinneaMatrixEncodingAttr::MatrixProperty::UnitDiagonal;
  if (p ==
      MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_LOWERTRIANGULAR)
    return LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular;
  if (p ==
      MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_UPPERTRIANGULAR)
    return LinneaMatrixEncodingAttr::MatrixProperty::UpperTriangular;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_SYMMETRIC)
    return LinneaMatrixEncodingAttr::MatrixProperty::Symmetric;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_SQUARE)
    return LinneaMatrixEncodingAttr::MatrixProperty::Square;
  if (p == MlirLinneaMatrixEncoding::MLIR_LINNEA_MATRIX_PROPERTY_SPD)
    return LinneaMatrixEncodingAttr::MatrixProperty::SPD;
  else
    return LinneaMatrixEncodingAttr::MatrixProperty::SPSD;
}

MlirAttribute mlirLinneaAttributeMatrixEncodingAttrGet(
    MlirContext ctx, intptr_t numProperties,
    MlirLinneaMatrixEncoding const *properties) {
  SmallVector<LinneaMatrixEncodingAttr::MatrixProperty> cppProperties;
  cppProperties.resize(numProperties);
  for (intptr_t i = 0; i < numProperties; i++)
    cppProperties[i] = getCppProperty(properties[i]);
  return wrap(LinneaMatrixEncodingAttr::get(unwrap(ctx), cppProperties));
}

bool mlirTypeIsLinneaMatrixType(MlirType type) {
  return unwrap(type).isa<MatrixType>();
}

MlirType mlirLinneaMatrixTypeGet(MlirContext ctx, MlirAttribute attr,
                                 intptr_t rank, const int64_t *shape,
                                 MlirType elementType) {
  return wrap(MatrixType::get(
      unwrap(ctx), unwrap(attr).cast<LinneaMatrixEncodingAttr>(),
      llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType)));
}

bool mlirTypeIsLinneaTermType(MlirType type) {
  return unwrap(type).isa<TermType>();
}

MlirType mlirLinneaTermTypeGet(MlirContext ctx) {
  return wrap(TermType::get(unwrap(ctx)));
}
