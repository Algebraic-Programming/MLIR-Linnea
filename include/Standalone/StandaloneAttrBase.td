#ifndef STANDALONE_ATTRDEFS
#define STANDALONE_ATTRDEFS

include "StandaloneDialect.td"

// All of the Standalone attributes will extend this class.
class Standalone_Attr<string name,
                      list<Trait> traits = []>
        : AttrDef<Standalone_Dialect, name, traits>;

// Standalone encoding attribute.
def StandaloneMatrixEncodingAttr : Standalone_Attr<"StandaloneMatrixEncoding"> {
  let mnemonic = "matrix_encoding";
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
          else if (strVal == "unitdiagonal")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::UnitDiagonal);
          else if (strVal == "lowertriangular")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::LowerTriangular);
          else if (strVal == "uppertriangular")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::UpperTriangular);
          else if (strVal == "symmetric")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::Symmetric);
          else if (strVal == "spd")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::SPD);
          else if (strVal == "spds")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::SPSD);
          else if (strVal == "identity")
            mt.push_back(StandaloneMatrixEncodingAttr::MatrixType::Identity);
          else {
            $_parser.emitError($_parser.getNameLoc(),
                               "unexpected matrix type") << strVal;
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
      UpperTriangular,
      Symmetric,
      SPD,
      SPSD,
      Identity
    };
  }];
}


#endif // STANDALONE_ATTRDEFS
