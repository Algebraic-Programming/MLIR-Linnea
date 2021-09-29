// RUN: standalone-opt --split-input-file --verify-diagnostics %s 

func @bar(%arg0 : !linnea.matrix<"A",["lowerTri"],[32,32]>) -> !linnea.matrix<"A", ["general"], [32,32]> {
  %0 = linnea.symtrans %arg0 : !linnea.matrix<"A", ["lowerTri"], [32, 32]> -> !linnea.matrix<"A", ["general"], [32, 32]>  // expected-error {{input is lowerTriangular then output must be lowerTriangular}}
  return %0 : !linnea.matrix<"A", ["general"], [32, 32]>
}
  
