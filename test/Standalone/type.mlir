// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.matrix<"A", ["identity"], [32, 32]>)
func private @check_type(%arg0: !linnea.matrix<"A",["identity"],[32,32]>)
