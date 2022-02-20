//===- LinneaUtils.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace linnea {

LinneaMatrixEncodingAttr getLinneaMatrixEncoding(Type type) {
  if (auto ttp = type.dyn_cast<MatrixType>())
    return ttp.getProperty().dyn_cast_or_null<LinneaMatrixEncodingAttr>();
  return nullptr;
}

LinneaMatrixEncodingAttr getLinneaTensorEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<LinneaMatrixEncodingAttr>();
  return nullptr;
}

} // namespace linnea
} // namespace mlir
