// RUN: standalone-opt %s --canonicalize | FileCheck %s

#F = #linnea.matrix_encoding<{encodingType = ["fullrank"]}>

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !linnea.matrix<tensor<2x2xf64>,#linnea.matrix_encoding<{encodingType = ["fullrank"]}>>)
    func @bar(%arg0 : !linnea.matrix<tensor<2x2xf64>, #F>) {
      // CHECK: return
      %0 = linnea.inverse %arg0 : !linnea.matrix<tensor<2x2xf64>, #F>
      %1 = linnea.inverse %0 : !linnea.matrix<tensor<2x2xf64>, #F>
      return
    }
}
