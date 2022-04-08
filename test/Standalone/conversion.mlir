// RUN: standalone-opt %s --properties-propagation --linnea-func-type-conversion --convert-linnea-to-linalg | FileCheck %s

// CHECK-LABEL: func @bar(
func @bar(%arg0: !linnea.matrix<#linnea.property<["general"]>,[32,32], f32>, 
          %arg1: !linnea.matrix<#linnea.property<["general"]>,[32,32], f32>) {
  %0 = linnea.equation {
    // CHECK: %{{.*}} = linalg.generic 
    %1 = linnea.mul.high %arg0, %arg1 { semirings = "min-plus" } : 
      !linnea.matrix<#linnea.property<["general"]>,[32,32], f32>, 
      !linnea.matrix<#linnea.property<["general"]>,[32,32], f32> -> !linnea.term
    linnea.yield %1 : !linnea.term
  }
  return
}
