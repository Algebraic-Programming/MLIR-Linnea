//===- LinneaAttributes.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_ATTRIBUTES_H
#define LINNEA_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Standalone/LinneaAttrBase.h.inc"

bool hasSPDAttr(mlir::Type type);
bool hasLowerTriangularAttr(mlir::Type type);
bool hasUpperTriangularAttr(mlir::Type type);

#endif // LINNEA_ATTRIBUTES_H
