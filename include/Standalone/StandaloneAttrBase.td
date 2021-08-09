#ifndef STANDALONE_ATTRDEFS
#define STANDALONE_ATTRDEFS

#include "StandaloneDialect.td"

// All of the Standalone attributes will extend this class.
class Standalone_Attr<string name,
                      list<Trait> traits = []>
        : AttrDef<Standalone_Dialect, name, traits>;

// Standalone encoding attribute.
def StandaloneEncodingAttr : Standalone_Attr<"StandaloneMatrixEncoding" {
  let mnemonic = "matrix-encoding";
  let description = [{ TODO:add }];
  let parameters = (
    ins
    ArrayRefParameter<"StandaloneMatrixEncoding::Type",
                      "type of matrix encoding"
                      >: $encodingType
  );
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    enum class Type {
      Diagonal,
      Identity
    };
  }];
}


#endif // STANDALONE_ATTRDEFS
