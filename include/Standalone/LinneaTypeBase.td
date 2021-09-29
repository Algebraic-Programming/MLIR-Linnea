#ifndef LINNEA_TYPE_BASE
#define LINNEA_TYPE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "LinneaDialect.td"

class Linnea_Type<string name>
  : TypeDef<Linnea_Dialect, name>;

def LinneaSymbolicMatrix : Linnea_Type<"Matrix"> {
  let mnemonic = "matrix";

  let summary = "A 2-dimensional tensor";
  let description = [{
    A 2-dimensional tensor.
  }];
  let genVerifyDecl = 1;

  let parameters = (ins
    "std::string":$id,
    ArrayRefParameter<"MatrixType::MatrixProperty",
                      "type of matrix encoding"
                      >:$property,
    ArrayRefParameter<"int64_t", "dimensions"
                     >:$dims
  );

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    std::string id;
    if ($_parser.parseString(&id))
      return Type();
    if ($_parser.parseComma())
      return Type();
    ArrayAttr properties;
    if ($_parser.parseAttribute(properties))
      return Type();

    SmallVector<MatrixProperty, 4> mt;
    for (size_t i = 0, e = properties.size(); i < e; i++) {
      auto strAttr = properties[i].dyn_cast<StringAttr>();
      if (!strAttr) {
        $_parser.emitError($_parser.getNameLoc(),
                           "expect string attribute for matrix type");
        return Type();
      }
      auto strVal = strAttr.getValue();
      if (strVal == "general")
        mt.push_back(MatrixType::MatrixProperty::General);
      else if (strVal == "fullrank")
        mt.push_back(MatrixType::MatrixProperty::FullRank);
      else if (strVal == "diagonal")
        mt.push_back(MatrixType::MatrixProperty::Diagonal);
      else if (strVal == "unitdiagonal")
        mt.push_back(MatrixType::MatrixProperty::UnitDiagonal);
      else if (strVal == "lowertriangular")
        mt.push_back(MatrixType::MatrixProperty::LowerTriangular);
      else if (strVal == "uppertriangular")
        mt.push_back(MatrixType::MatrixProperty::UpperTriangular);
      else if (strVal == "symmetric")
        mt.push_back(MatrixType::MatrixProperty::Symmetric);
      else if (strVal == "spd")
        mt.push_back(MatrixType::MatrixProperty::SPD);
      else if (strVal == "spds")
        mt.push_back(MatrixType::MatrixProperty::SPSD);
      else if (strVal == "identity")
        mt.push_back(MatrixType::MatrixProperty::Identity);
      else {
        $_parser.emitError($_parser.getNameLoc(),
                           "unexpected matrix type: ") << strVal;
        return {};
      }
    }

    if ($_parser.parseComma())
      return Type();
    ArrayAttr dimensions;
    if ($_parser.parseAttribute(dimensions))
      return Type();
    
    SmallVector<int64_t, 4> dims; 
    for (size_t i = 0, e = dimensions.size(); i < e; i++) {
      auto intAttr = dimensions[i].dyn_cast<IntegerAttr>();
      if (!intAttr) {
        $_parser.emitError($_parser.getNameLoc(),
                           "expect int attribute for matrix type");
        return Type();
      }
      dims.push_back(intAttr.getInt());
    }

    if ($_parser.parseGreater())
      return Type();
    return MatrixType::getChecked([&$_parser] { 
      return $_parser.emitError($_parser.getCurrentLocation()); }, 
             $_ctxt, id, mt, dims);
  }];

 let extraClassDeclaration = [{
    enum class MatrixProperty {
      General,
      FullRank,
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

#endif // LINNEA_TYPE_BASE 
