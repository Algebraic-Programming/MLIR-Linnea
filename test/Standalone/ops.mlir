// RUN: standalone-opt --split-input-file %s | standalone-opt | FileCheck %s

#F = #linnea.matrix<{p = ["fullrank"]}>

// CHECK-LABEL: func @bar(
func @bar(%arg0 : tensor<2x2xf64, #F>, %arg1 : tensor<2x3xf64>, %arg2 : tensor<3x2xf64>) {
  // CHECK: %{{.*}} = linnea.inverse %{{.*}} : tensor<2x2xf64, #linnea.matrix<{p = ["fullrank"]}>>
  %res = linnea.inverse %arg0 : tensor<2x2xf64, #F>
  %chain = linnea.mul %arg0, %arg1, %arg2 : tensor<2x2xf64, #F>, tensor<2x3xf64>, tensor<3x2xf64> -> tensor<2x2xf64>
  return
}

// -----

// CHECK-LABEL: func @bar
func @bar(%arg0 : tensor<?x?xf64>, %arg1 : tensor<?x?xf64>, %arg2 : tensor<?x?xf64>) -> tensor<?x?xf64> {
  // CHECK: %{{.*}} = linnea.mul %{{.*}} : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %chain = linnea.mul %arg0, %arg1, %arg2 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %chain : tensor<?x?xf64>
}

// -----
// CHECK-LABEL: func @bar
func @bar(%arg0 : !linnea.matrix<"A",["general"],[32,32]>) -> !linnea.matrix<"A", ["general"], [32,32]> {
  %0 = linnea.symtrans %arg0 : !linnea.matrix<"A", ["general"], [32, 32]> -> !linnea.matrix<"A", ["general"], [32, 32]>  
  return %0 : !linnea.matrix<"A", ["general"], [32, 32]>
}
  
