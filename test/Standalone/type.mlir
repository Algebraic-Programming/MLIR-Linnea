// RUN: standalone-opt --split-input-file %s | standalone-opt | FileCheck %s

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.matrix<["identity"], [32, 32]>)
func private @check_type(%arg0: !linnea.matrix<["identity"],[32,32]>)

// -----

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.term
func private @check_type(%arg0: !linnea.term)
