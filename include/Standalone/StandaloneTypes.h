#ifndef STANDALONE_TYPES_H
#define STANDALONE_TYPES_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "Standalone/StandaloneTypeBase.h.inc"

#endif // STANDALONE_TYPES_H
