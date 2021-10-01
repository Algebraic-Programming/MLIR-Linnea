#ifndef LINNEA_ATTRIBUTES_H
#define LINNEA_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Standalone/LinneaAttrBase.h.inc"

bool isSPD(mlir::Type type);
bool isLowerTriangular(mlir::Type type);
bool isUpperTriangular(mlir::Type type);

#endif // LINNEA_ATTRIBUTES_H
