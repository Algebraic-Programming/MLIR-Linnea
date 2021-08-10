#ifndef STANDALONE_TYPE_BASE
#define STANDALONE_TYPE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "StandaloneDialect.td"

class Standalone_Type<string name>
  : TypeDef<Standalone_Dialect, name>;

def Standalone_Matrix : Standalone_Type<"Matrix"> {
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

#endif // STANDALONE_TYPE_BASE 
