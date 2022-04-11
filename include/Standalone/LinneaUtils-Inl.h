//===- LinneaUtils-Inl.h - Linnea utils -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linnea {

template <typename FOpTy, typename IOpTy>
static Value buildBinaryOpFromValues(OpBuilder builder, Value left, Value right,
                                     Location loc, Type t) {
  if (isMLIRFloatType(t))
    return builder.create<FOpTy>(loc, left, right);
  else if (isMLIRIntType(t))
    return builder.create<IOpTy>(loc, left, right);
  llvm_unreachable("unsupported type");
}

} // namespace linnea
} // namespace mlir
