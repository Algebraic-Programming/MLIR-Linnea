#ifndef MLIR_LINNEA_TYPE_CONVERTER_H
#define MLIR_LINNEA_TYPE_CONVERTER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace linnea {

class LinneaTypeConverter : public TypeConverter {
public:
  LinneaTypeConverter() = default;
};

} // end namespace linnea
} // namespace mlir
#endif
