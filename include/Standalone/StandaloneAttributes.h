#ifndef STANDALONE_ATTRIBUTES_H
#define STANDALONE_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Standalone/StandaloneAttrBase.h.inc"

#endif // STANDALONE_ATTRIBUTES_H
