// RUN: standalone-opt %s | FileCheck %s 
// XFAIL: *
func @foo(%arg0: !linnea.matrix<#linnea.property<["lowerTri"]>, 
                                                 [30, 30], f32>,
          %arg1: !linnea.matrix<#linnea.property<["lowerTri"]>, 
                                                 [30, 30], f32>,
          %arg3: f32) -> !linnea.term {

  %1 = linnea.fill(%arg3, %arg0) : 
    f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [30, 30], f32>
  %2 = linnea.fill(%arg3, %arg1) :
    f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [30, 30], f32>

  %0 = linnea.equation {
    %3 = linnea.mul.high %1, %2: 
        !linnea.matrix<#linnea.property<["lowerTri"]>, [30, 30], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [30, 30], f32> -> !linnea.term
    linnea.yield %3 : !linnea.term
  } 
  return %0 : !linnea.term
}
