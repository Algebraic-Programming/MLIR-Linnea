// RUN: standalone-opt %s --linnea-func-type-conversion | FileCheck %s
// CHECK-LABEL: func @bar(
// CHECK-SAME: %[[ARG:.*]]: tensor<30x35xf32, #linnea.property<["fullrank"]>>) -> tensor<30x35xf32, #linnea.property<["fullrank"]>> {
// CHECK: return %[[ARG]] : tensor<30x35xf32, #linnea.property<["fullrank"]>>
func @bar(%arg0 : !linnea.matrix<#linnea.property<["fullrank"]>, [30, 35], f32>) -> 
    !linnea.matrix<#linnea.property<["fullrank"]>, [30, 35], f32> {
  return %arg0 : !linnea.matrix<#linnea.property<["fullrank"]>, [30, 35], f32>
}
