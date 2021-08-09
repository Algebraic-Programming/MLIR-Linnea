#ifndef STANDALONE_ATTRDEFS
#define STANDALONE_ATTRDEFS

include "StandaloneDialect.td"

// All of the Standalone attributes will extend this class.
class Standalone_Attr<string name,
                      list<Trait> traits = []>
        : AttrDef<Standalone_Dialect, name, traits>;

// Standalone encoding attribute.
def StandaloneMatrixEncodingAttr : Standalone_Attr<"StandaloneMatrixEncoding"> {
  let mnemonic = "matrix";
  let description = [{ TODO:add }];
  let parameters = (
    ins
    ArrayRefParameter<"StandaloneMatrixEncodingAttr::MatrixType",
                      "type of matrix encoding"
                      >: $encodingType
  );
  let genVerifyDecl = 1;
  
  let parser = [{
    if ($_parser.parseLess())
      return {};
    DictionaryAttr dict;
    if ($_parser.parseAttribute(dict))
      return {};
    if ($_parser.parseGreater())
      return {};
    SmallVector<MatrixType, 4> mt;

    for (const NamedAttribute &attr : dict) {
      if (attr.first == "encodingType") {
        auto arrayAttr = attr.second.dyn_cast<ArrayAttr>();
        if (!arrayAttr) {
          $_parser.emitError($_parser.getNameLoc(), 
                             "expect an array for matrix encoding");
          return {};
        }
        for (size_t i = 0, e = arrayAttr.size(); i < e; i++) {
          auto strAttr = arrayAttr[i].dyn_cast<StringAttr>();
          if (!strAttr) {
            $_parser.emitError($_parser.getNameLoc(),
                               "expect string attribute for matrix type");
            return {};
          }
          auto strVal = strAttr.getValue();
          if (strVal == "diagonal")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::Diagonal);
          else {
            $_parser.emitError($_parser.getNameLoc(),
                               "unexpected matrix type");
            return {};
          }
        }
      }
    }
    return StandaloneMatrixEncodingAttr::getChecked([&$_parser] {
      return $_parser.emitError($_parser.getCurrentLocation()); },
      $_ctxt, mt);
  }];

  //let printer = [{
  //  $_printer << "ecoding matrix";  
  //]};

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
