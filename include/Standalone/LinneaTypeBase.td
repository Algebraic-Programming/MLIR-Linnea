#ifndef LINNEA_TYPE_BASE
#define LINNEA_TYPE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "LinneaDialect.td"

class Linnea_Type<string name>
  : TypeDef<Linnea_Dialect, name>;

def Linnea_Matrix : Linnea_Type<"Matrix"> {
  let mnemonic = "matrix";

  let summary = "A 2-dimensional tensor";
  let description = [{
    A 2-dimensional tensor.
  }];
  let genVerifyDecl = 1;

  let parameters = (ins
    // TODO: here I would like a ranked tensor.
    "TensorType":$param,
    "Attribute":$encoding
  );

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    Type tensor;
    if ($_parser.parseType(tensor))
      return Type();
    if ($_parser.parseComma())
      return Type();
    Attribute attribute;
    if ($_parser.parseAttribute(attribute))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    auto tensorType = tensor.dyn_cast<TensorType>();
    if (!tensorType)
      return Type();
    return MatrixType::getChecked([&$_parser] { 
      return $_parser.emitError($_parser.getCurrentLocation()); }, 
             $_ctxt, tensorType, attribute);
  }];

  //let printer = [{
  //
  //}];
      
}

#endif // LINNEA_TYPE_BASE 
