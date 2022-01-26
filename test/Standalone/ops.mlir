// RUN: standalone-opt --split-input-file %s | standalone-opt | FileCheck %s

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.term, %arg1: !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32>) {
  // CHECK: %{{.*}} = linnea.mul.low %{{.*}}
  %0 = linnea.mul.low %arg0, %arg1 : !linnea.term, !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32> -> !linnea.term
  return
}

// -----

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.term, %arg1: !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32>) {
  %0 = linnea.equation {
    // CHECK: %{{.*}} = linnea.mul.high %{{.*}}
    %1 = linnea.mul.high %arg0, %arg1 : !linnea.term, !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32> -> !linnea.term
    linnea.yield %1 : !linnea.term
  }
  return
}

// -----

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32>) {
  // CHECK: %{{.*}} = linnea.equation
  %0 = linnea.equation {
      // CHECK: %{{.*}} = linnea.transpose %{{.*}}
      %1 = linnea.transpose %arg0 : !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32> -> !linnea.term
      // CHECK: linnea.yield %{{.*}}
      linnea.yield %1 : !linnea.term
  }
  return
}

// -----

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32>, %arg1: f32) {
  // CHECK: %{{.*}} = linnea.fill
  linnea.fill(%arg1, %arg0) : f32, !linnea.matrix<#linnea.property<["identity"]>,[32,32], f32>
  return
}
