//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::standalone;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Standalone/StandaloneTypeBase.cpp.inc"

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/StandaloneOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Standalone/StandaloneTypeBase.cpp.inc"
    >();
}

mlir::Type StandaloneDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef ref;
  if (parser.parseKeyword(&ref))
    return Type();
  Type res;
  auto parsed = generatedTypeParser(getContext(), parser, ref, res);
  if (parsed.hasValue() && succeeded(parsed.getValue()))
    return res;
  return Type();
}

void StandaloneDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  auto wasPrinted = generatedTypePrinter(type, printer);
  assert(succeeded(wasPrinted));
}
