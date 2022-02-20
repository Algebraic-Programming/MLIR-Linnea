#include "Standalone-c/LinneaDialect.h"

#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaOps.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace mlir::linnea;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linnea, linnea,
                                      mlir::linnea::LinneaDialect)

bool mlirAttributeIsLinneaMatrixEncodingAttr(MlirAttribute attr) {
  return unwrap(attr).isa<LinneaMatrixEncodingAttr>();
}

MlirAttribute mlirLinneaAttributeMatrixEncodingAttrGet(
    MlirContext ctx, intptr_t numProperties,
    MlirLinneaMatrixEncoding const *properties) {
  SmallVector<LinneaMatrixEncodingAttr::MatrixProperty> cppProperties;
  cppProperties.resize(numProperties);
  for (intptr_t i = 0; i < numProperties; i++)
    cppProperties[i] =
        static_cast<LinneaMatrixEncodingAttr::MatrixProperty>(properties[i]);
  return wrap(LinneaMatrixEncodingAttr::get(unwrap(ctx), cppProperties));
}

bool mlirTypeIsLinneaMatrixType(MlirType type) {
  return unwrap(type).isa<MatrixType>();
}

MlirType mlirLinneaMatrixTypeGet(MlirContext ctx, MlirAttribute attr,
                                 intptr_t rank, const int64_t *shape,
                                 MlirType elementType) {
  return wrap(MatrixType::get(
      unwrap(ctx), unwrap(attr).cast<LinneaMatrixEncodingAttr>(),
      llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType)));
}

bool mlirTypeIsLinneaTermType(MlirType type) {
  return unwrap(type).isa<TermType>();
}

MlirType mlirLinneaTermTypeGet(MlirContext ctx) {
  return wrap(TermType::get(unwrap(ctx)));
}
