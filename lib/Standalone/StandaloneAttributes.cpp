#include "Standalone/StandaloneAttributes.h"
#include "Standalone/StandaloneDialect.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::standalone;

Attribute StandaloneMatrixEncodingAttr::parse(MLIRContext *context,
                                              DialectAsmParser &parser,
                                              Type type) {
  return {};
}

void StandaloneMatrixEncodingAttr::print(DialectAsmPrinter &printer) const {
  printer << "encoding-matrix";
}
