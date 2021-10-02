// RUN: standalone-opt %s --split-input-file --canonicalize | FileCheck %s
// XFAILS: *

// CHECK-LABEL: func @bar(%{{.*}}: !linnea.matrix<["square"], [32,64]>)
func @bar(%arg0 : !linnea.matrix<["square"], [32,64]>) {
  // CHECK: return
  %0 = linnea.inverse %arg0 : !linnea.matrix<["square"], [32,64]> -> !linnea.matrix<["square"], [64,32]>
  %1 = linnea.inverse %0 : !linnea.matrix<["square"],[64,32]> -> !linnea.matrix<["square"], [32,64]>
  return
}

// -----

func @bar(%A : !linnea.matrix<["square", "fullrank"], [2,2]>, %S : !linnea.matrix<["spd"], [2,2]>) -> !linnea.matrix<["spd"], [2,2]>{
  %0 = linnea.inverse %S : !linnea.matrix<["spd"], [2,2]> -> !linnea.matrix<["spd"], [2,2]>
  %1 = linnea.transpose %A : !linnea.matrix<["square", "fullrank"], [2,2]> -> !linnea.matrix<["square", "fullrank"], [2,2]>
  %2 = linnea.mul %1, %0, %A : !linnea.matrix<["square", "fullrank"], [2,2]>, !linnea.matrix<["spd"], [2,2]>, !linnea.matrix<["square", "fullrank"], [2,2]> -> !linnea.matrix<["spd"], [2,2]>
  return %2 : !linnea.matrix<["spd"], [2,2]>
}
