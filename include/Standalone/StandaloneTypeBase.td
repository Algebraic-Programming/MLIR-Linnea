#ifndef STANDALONE_TYPE_BASE
#define STANDALONE_TYPE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "StandaloneDialect.td"

class Standalone_Type<string name, list<Trait> traits = []/*,
                      string baseCppClass = "::mlir::Type"*/>
  : TypeDef<Standalone_Dialect, name, traits/*, baseCppClass*/>;

def Standalone_Matrix : Standalone_Type<"Matrix", []> {
  let mnemonic = "matrix";
  let summary = "2-dimensional tensor";
  let description = [{
    A multiset type. Similar to List Type, but without ordering and indexing.
  }];

  let parameters = (ins
    //ArrayRefParameter<"int64_t">:$shape,
    //"Type":$elementType,
    "TensorType":$param,
    "Attribute":$encoding
  );
  
  
/*
  let builders = [
    TypeBuilderWithInferredContext<(ins
      //"ArrayRef<int64_t>":$shape,
      //"Type":$elementType,
      "TensorType":$param,
      CArg<"Attribute", "{}">:$encoding
    ), [{
      return $_get(param.getContext(), shape, elementType, encoding);
    }]>
  ];
*/
  //let genVerifyDecl = 1;
}

#endif // STANDALONE_TYPE_BASE 
