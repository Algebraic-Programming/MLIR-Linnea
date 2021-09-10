// RUN: standalone-opt %s | standalone-opt | FileCheck %s
// XFAIL: *
#I = #linnea.matrix_encoding<{encodingType = ["identity"]}>

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.matrix<tensor<32x32xf64>,#linnea.matrix_encoding<{encodingType = ["identity"]}>>)
func private @check_type(%arg0: !linnea.matrix<tensor<32x32xf64>, #I>)
