// RUN: standalone-opt %s | standalone-opt | FileCheck %s

#F = #linnea.matrix_encoding<{encodingType = ["fullrank"]}>

module {
    // CHECK-LABEL: func @bar(
    func @bar(%arg0 : !linnea.matrix<tensor<2x2xf64>, #F>) {
      // CHECK: %{{.*}} = linnea.inverse %{{.*}} : !linnea.matrix<tensor<2x2xf64>,#linnea.matrix_encoding<{encodingType = ["fullrank"]}>>
      %res = linnea.inverse %arg0 : !linnea.matrix<tensor<2x2xf64>, #F>
      return
    }
}
