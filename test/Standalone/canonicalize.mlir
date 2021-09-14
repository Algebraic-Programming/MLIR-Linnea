// RUN: standalone-opt %s --split-input-file --canonicalize | FileCheck %s

#F = #linnea.matrix<{p = ["general"]}>

// CHECK-LABEL: func @bar(%{{.*}}: tensor<2x2xf64, #linnea.matrix<{p = ["general"]}>>)
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

#S = #linnea.matrix<{p = ["spd"]}>
// CHECK-LABEL: func @bar
// CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: tensor<2x2xf64>
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: tensor<2x3xf64, #linnea.matrix<{p = ["spd"]}>>
// CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: tensor<3x2xf64>
func @bar(%arg0 : tensor<2x2xf64>, %arg1 : tensor<2x3xf64, #S>, %arg2 : tensor<3x2xf64>) -> tensor<2x2xf64> {
  // CHECK: %[[v0:.*]] = linnea.cholesky %[[arg1]] : tensor<2x3xf64, #linnea.matrix<{p = ["spd"]}>>
  // CHECK: %[[v1:.*]] = linnea.transpose %[[v0]] : tensor<2x3xf64, #linnea.matrix<{p = ["upperTri"]}>>
  // CHECK: %[[v2:.*]] = linnea.mul %[[v0]], %[[v1]] : tensor<2x3xf64, #linnea.matrix<{p = ["upperTri"]}>>, tensor<2x3xf64, #linnea.matrix<{p = ["lowerTri"]}>> -> tensor<2x3xf64>
  // CHECK: %[[v3:.*]] = linnea.inverse %[[v2]] : tensor<2x3xf64>
  // CHECK: %[[v4:.*]] = linnea.mul %[[arg0]], %[[v3]], %[[arg2]] : tensor<2x2xf64>, tensor<2x3xf64>, tensor<3x2xf64> -> tensor<2x2xf64>
  %0 = linnea.inverse %arg1 : tensor<2x3xf64, #S>
  %1 = linnea.mul %arg0, %0, %arg2 : tensor<2x2xf64>, tensor<2x3xf64, #S>, tensor<3x2xf64> -> tensor<2x2xf64>
  // CHECK: return %[[v4]] : tensor<2x2xf64>
  return %1 : tensor<2x2xf64>
}
