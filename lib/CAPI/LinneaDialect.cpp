#include "Standalone-c/LinneaDialect.h"

#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaOps.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linnea, linnea,
                                      mlir::linnea::LinneaDialect)
