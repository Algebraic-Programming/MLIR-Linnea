// RUN: standalone-opt --split-input-file %s | standalone-opt | FileCheck %s

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.term, %arg1: !linnea.matrix<["identity"],[32,32]>) {
  // CHECK: %{{.*}} = linnea.mul %{{.*}}
  %0 = linnea.mul %arg0, %arg1 : !linnea.term, !linnea.matrix<["identity"],[32,32]> -> !linnea.term
  return
}

// -----

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.term, %arg1: !linnea.matrix<["identity"],[32,32]>) {
  // CHECK: %{{.*}} = linnea.equ %{{.*}}
  %0 = linnea.equ %arg1: !linnea.matrix<["identity"],[32,32]> -> !linnea.term {
    ^bb0(%0: !linnea.matrix<["identity"],[32,32]>):
      // CHECK: %{{.*}} = linnea.transpose %{{.*}}
      %1 = linnea.transpose %0 : !linnea.matrix<["identity"],[32,32]> -> !linnea.term
      // CHECK: linnea.yield %{{.*}}
      linnea.yield %1 : !linnea.term
  }
  return
}

// -----

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.term, %arg1: !linnea.matrix<["identity"],[32,32]>) {
  // CHECK: %{{.*}} = linnea.equ %{{.*}}
  %0 = linnea.equ %arg1: !linnea.matrix<["identity"],[32,32]> -> !linnea.term {
    ^bb0(%0: !linnea.matrix<["identity"],[32,32]>):
      // CHECK: linnea.yield %{{.*}}
      linnea.yield %0 : !linnea.matrix<["identity"],[32,32]>
  }
  return
}
 
