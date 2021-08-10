#ifndef LINNEA_ATTRDEFS
#define LINNEA_ATTRDEFS

include "LinneaDialect.td"

// All of the Linnea attributes will extend this class.
class Linnea_Attr<string name,
                      list<Trait> traits = []>
        : AttrDef<Linnea_Dialect, name, traits>;

// Linnea encoding attribute.
def LinneaMatrixEncodingAttr : Linnea_Attr<"LinneaMatrixEncoding"> {
  let mnemonic = "matrix_encoding";
  let description = [{ TODO:add }];
  let parameters = (
    ins
    ArrayRefParameter<"LinneaMatrixEncodingAttr::MatrixType",
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
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::Diagonal);
          else if (strVal == "unitdiagonal")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::UnitDiagonal);
          else if (strVal == "lowertriangular")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::LowerTriangular);
          else if (strVal == "uppertriangular")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::UpperTriangular);
          else if (strVal == "symmetric")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::Symmetric);
          else if (strVal == "spd")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::SPD);
          else if (strVal == "spds")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::SPSD);
          else if (strVal == "identity")
            mt.push_back(LinneaMatrixEncodingAttr::MatrixType::Identity);
          else {
            $_parser.emitError($_parser.getNameLoc(),
                               "unexpected matrix type") << strVal;
            return {};
          }
        }
      }
    }
    return LinneaMatrixEncodingAttr::getChecked([&$_parser] {
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


#endif // LINNEA_ATTRDEFS
