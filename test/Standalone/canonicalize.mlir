// RUN: standalone-opt %s --properties-propagation --canonicalize | FileCheck %s

// CHECK-LABEL: @bar
// CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: !linnea.matrix<#linnea.property<["square"]>, [32, 32], f32>
func @bar(%arg0 : !linnea.matrix<#linnea.property<["square"]>, [32,32], f32>) -> !linnea.term {
  
  %0 = linnea.equation {
    %1 = linnea.inverse.high %arg0 : !linnea.matrix<#linnea.property<["square"]>, [32,32], f32> 
      -> !linnea.matrix<#linnea.property<["square"]>, [32,32], f32>
    %2 = linnea.inverse.high %1 : !linnea.matrix<#linnea.property<["square"]>,[32,32], f32> 
      -> !linnea.term
    linnea.yield %2 : !linnea.term
  }
  // CHECK: return %[[arg0:[a-zA-Z0-9]+]] : !linnea.matrix<#linnea.property<["square"]>, [32, 32], f32>
  return %0 : !linnea.term
}
