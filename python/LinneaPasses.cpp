//===- LinneaPasses.cpp - Pybind module for Linnea passes------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/LinneaDialect.h"
#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirLinneaPasses, m) {
  m.doc() = "MLIR Linnea Dialect Passes";

  // Register all SparseTensor passes on load.
  mlirRegisterLinneaPasses();
}
