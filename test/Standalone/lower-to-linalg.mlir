// RUN: standalone-opt %s | FileCheck %s 
// XFAIL: *
func @foo(%arg0: !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32>, 
          %arg1: !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32>) -> !linnea.matrix<#linnea.property<["lowerTri", "square"]>, [30, 30], f32> {
  %0 = linnea.mul %arg0, %arg1 : !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32>, !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32> -> !linnea.matrix<#linnea.property<["lowerTri", "square"]>, [30, 30], f32>
  return %0 : !linnea.matrix<#linnea.property<["lowerTri", "square"]>, [30, 30], f32>
}
