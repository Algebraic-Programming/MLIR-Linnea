// RUN: standalone-opt %s --split-input-file --convert-linnea-to-linalg | FileCheck %s
// XFAIL: *
// CHECK-LABEL: func @bar
func @bar(%arg0 : tensor<2x2xf64>, %arg1 : tensor<2x3xf64>, %arg2 : tensor<3x2xf64>) -> tensor<2x2xf64> {
  %chain = linnea.mul %arg0, %arg1, %arg2 : tensor<2x2xf64>, tensor<2x3xf64>, tensor<3x2xf64> -> tensor<2x2xf64>
  return %chain : tensor<2x2xf64>
}

// -----

// CHECK-LABEL: func @bar
func @bar(%arg0 : tensor<?x?xf64>, %arg1 : tensor<?x?xf64>, %arg2 : tensor<?x?xf64>) -> tensor<?x?xf64> {
  %chain = linnea.mul %arg0, %arg1, %arg2 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %chain : tensor<?x?xf64>
}

// -----

// CHECK-LABEL: @bar
// CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: tensor<30x35xf64>
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: tensor<35x15xf64>
// CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: tensor<15x5xf64>
// CHECK-SAME: %[[arg3:[a-zA-Z0-9]+]]: tensor<5x10xf64>
// CHECK-SAME: %[[arg4:[a-zA-Z0-9]+]]: tensor<10x20xf64>
// CHECK-SAME: %[[arg5:[a-zA-Z0-9]+]]: tensor<20x25xf64>
func @bar(%arg0 : tensor<30x35xf64>, %arg1 : tensor<35x15xf64>, %arg2 : tensor<15x5xf64>, 
          %arg3 : tensor<5x10xf64>, %arg4 : tensor<10x20xf64>, %arg5 : tensor<20x25xf64>) -> tensor<30x25xf64> {
  // CHECK: %[[v0:.*]] = linalg.init_tensor [35, 5] : tensor<35x5xf64>
  // CHECK: %[[v1:.*]] = linalg.matmul ins(%[[arg1]], %[[arg2]] : tensor<35x15xf64>, tensor<15x5xf64>) outs(%[[v0]] : tensor<35x5xf64>) -> tensor<35x5xf64>
  // CHECK: %[[v2:.*]] = linalg.init_tensor [30, 5] : tensor<30x5xf64>
  // CHECK: %[[v3:.*]] = linalg.matmul ins(%[[arg0]], %[[v1]] : tensor<30x35xf64>, tensor<35x5xf64>) outs(%[[v2]] : tensor<30x5xf64>) -> tensor<30x5xf64>
  // CHECK: %[[v4:.*]] = linalg.init_tensor [5, 20] : tensor<5x20xf64>
  // CHECK: %[[v5:.*]] = linalg.matmul ins(%[[arg3]], %[[arg4]] : tensor<5x10xf64>, tensor<10x20xf64>) outs(%[[v4]] : tensor<5x20xf64>) -> tensor<5x20xf64>
  // CHECK: %[[v6:.*]] = linalg.init_tensor [5, 25] : tensor<5x25xf64>
  // CHECK: %[[v7:.*]] = linalg.matmul ins(%[[v5]], %[[arg5]] : tensor<5x20xf64>, tensor<20x25xf64>) outs(%[[v6]] : tensor<5x25xf64>) -> tensor<5x25xf64>
  // CHECK: %[[v8:.*]] = linalg.init_tensor [30, 25] : tensor<30x25xf64>
  // CHECK: %[[v9:.*]] = linalg.matmul ins(%[[v3]], %[[v7]] : tensor<30x5xf64>, tensor<5x25xf64>) outs(%[[v8]] : tensor<30x25xf64>) -> tensor<30x25xf64>
  %chain = linnea.mul %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<30x35xf64>, tensor<35x15xf64>, 
                                                                 tensor<15x5xf64>, tensor<5x10xf64>,
                                                                 tensor<10x20xf64>, tensor<20x25xf64> -> tensor<30x25xf64>
  // CHECK: return %[[v9]] : tensor<30x25xf64>
  return %chain : tensor<30x25xf64>
}
