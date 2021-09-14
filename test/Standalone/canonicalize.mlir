// RUN: standalone-opt %s --split-input-file --canonicalize | FileCheck %s

#F = #linnea.matrix_encoding<{encodingType = ["general"]}>

// CHECK-LABEL: func @bar(%{{.*}}: tensor<2x2xf64, #linnea.matrix_encoding<{encodingType = ["general"]}>>)
func @bar(%arg0 : tensor<2x2xf64, #F>) {
  // CHECK: return
  %0 = linnea.inverse %arg0 : tensor<2x2xf64, #F>
  %1 = linnea.inverse %0 : tensor<2x2xf64, #F>
  return
}

// -----

// CHECK-LABEL: func @bar(%{{.*}}: tensor<2x2xf64>)
func @bar(%arg0 : tensor<2x2xf64>) {
  // CHECK: return
  %0 = linnea.inverse %arg0 : tensor<2x2xf64>
  %1 = linnea.inverse %0 : tensor<2x2xf64>
  return
}

// -----

#S = #linnea.matrix_encoding<{encodingType = ["spd"]}>
// CHECK-LABEL: loren
func @bar(%A : tensor<2x2xf64>, %B : tensor<2x2xf64, #S>, %C : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %0 = linnea.inverse %B : tensor<2x2xf64, #S>
  %1 = linnea.mul %A, %0, %C : tensor<2x2xf64>, tensor<2x2xf64, #S>, tensor<2x2xf64> -> tensor<2x2xf64>
  return %1 : tensor<2x2xf64>
}
