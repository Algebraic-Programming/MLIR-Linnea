// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: func private @check_encoding(
// CHECK-SAME: tensor<32x32xf64, #standalone.matrix_encoding<{encodingType = ["identity"]}>>)
func private @check_encoding(tensor<32x32xf64, #standalone.matrix_encoding<{encodingType = ["identity"]}>>)
