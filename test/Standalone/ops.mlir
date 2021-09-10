// RUN: standalone-opt %s | standalone-opt | FileCheck %s

#F = #linnea.matrix_encoding<{encodingType = ["fullrank"]}>

module {
    // CHECK-LABEL: func @bar(
    func @bar(%arg0 : tensor<2x2xf64, #F>, %arg1 : tensor<2x3xf64>, %arg2 : tensor<3x2xf64>) {
      // CHECK: %{{.*}} = linnea.inverse %{{.*}} : tensor<2x2xf64, #linnea.matrix_encoding<{encodingType = ["fullrank"]}>>
      %res = linnea.inverse %arg0 : tensor<2x2xf64, #F>
      %chain = linnea.mul %arg0, %arg1, %arg2 : tensor<2x2xf64, #F>, tensor<2x3xf64>, tensor<3x2xf64> -> tensor<2x2xf64>
      return
    }
}
