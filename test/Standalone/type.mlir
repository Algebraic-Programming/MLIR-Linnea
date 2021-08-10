// RUN: standalone-opt %s | standalone-opt | FileCheck %s

#I = #standalone.matrix_encoding<{encodingType = ["identity"]}>

// CHECK: func private @check_type(
// CHECK-SAME: !standalone.matrix<tensor<32x32xf64>,#standalone.matrix_encoding<{encodingType = ["identity"]}>>)
func private @check_type(%arg0: !standalone.matrix<tensor<32x32xf64>, #I>)
