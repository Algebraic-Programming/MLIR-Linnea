#ifndef STANDALONE_ATTRDEFS
#define STANDALONE_ATTRDEFS

include "StandaloneDialect.td"

// All of the Standalone attributes will extend this class.
class Standalone_Attr<string name,
                      list<Trait> traits = []>
        : AttrDef<Standalone_Dialect, name, traits>;

// Standalone encoding attribute.
def StandaloneMatrixEncodingAttr : Standalone_Attr<"StandaloneMatrixEncoding"> {
  let mnemonic = "matrix-encoding";
  let description = [{ TODO:add }];
  let parameters = (
    ins
    ArrayRefParameter<"StandaloneMatrixEncodingAttr::MatrixType",
                      "type of matrix encoding"
                      >: $encodingType
  );
  let extraClassDeclaration = [{
    enum class MatrixType {
      Diagonal,
      UnitDiagonal,
      LowerTriangular,
      UpperTrinagular,
      Symmetric,
      SPD,
      SPSD,
      Identity
    };
  }];
}


#endif // STANDALONE_ATTRDEFS
