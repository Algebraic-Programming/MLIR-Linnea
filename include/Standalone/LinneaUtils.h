//===- LinneaUtils.h - Linnea utils -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_UTILS_H
#define LINNEA_UTILS_H

namespace mlir {
class Type;
};

namespace mlir::linnea {
class LinneaMatrixEncodingAttr;
};

namespace mlir {
namespace linnea {

LinneaMatrixEncodingAttr getLinneaTensorEncoding(Type type);
LinneaMatrixEncodingAttr getLinneaMatrixEncoding(Type type);

} // namespace linnea
} // namespace mlir

#endif
