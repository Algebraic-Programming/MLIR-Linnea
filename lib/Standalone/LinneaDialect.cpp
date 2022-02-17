//===- LinneaDialect.cpp - Linnea dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaOpsDialect.cpp.inc"
#include "Standalone/LinneaTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::linnea;

//===----------------------------------------------------------------------===//
// Linnea dialect.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Standalone/LinneaTypeBase.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Standalone/LinneaAttrBase.cpp.inc"

void LinneaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/LinneaOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Standalone/LinneaTypeBase.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Standalone/LinneaAttrBase.cpp.inc"
      >();
}
/*
Type LinneaDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef ref;
  if (parser.parseKeyword(&ref))
    return Type();
  Type res;
  auto parsed = generatedTypeParser(parser, ref, res);
  if (parsed.hasValue() && succeeded(parsed.getValue()))
    return res;
  return Type();
}

void LinneaDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto wasPrinted = generatedTypePrinter(type, printer);
  assert(succeeded(wasPrinted));
}

Attribute LinneaDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
  if (parseResult.hasValue())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown linnea attribute: ")
      << attrTag;
  return Attribute();
}

void LinneaDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
*/
