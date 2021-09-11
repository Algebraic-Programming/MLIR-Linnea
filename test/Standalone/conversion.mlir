// RUN: standalone-opt %s --split-input-file --convert-linnea-to-linalg | FileCheck %s

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

// CHECK-LABEL: func @bar
func @bar(%arg0 : tensor<30x35xf64>, %arg1 : tensor<35x15xf64>, %arg2 : tensor<15x5xf64>, 
          %arg3 : tensor<5x10xf64>, %arg4 : tensor<10x20xf64>, %arg5 : tensor<20x25xf64>) -> tensor<30x25xf64> {
  %chain = linnea.mul %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<30x35xf64>, tensor<35x15xf64>, 
                                                                 tensor<15x5xf64>, tensor<5x10xf64>,
                                                                 tensor<10x20xf64>, tensor<20x25xf64> -> tensor<30x25xf64>
  return %chain : tensor<30x25xf64>
}
