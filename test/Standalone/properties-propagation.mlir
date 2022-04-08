// RUN: standalone-opt %s --split-input-file --properties-propagation | FileCheck %s
module {
  // CHECK: entry
  func @entry() {

    %c5 = arith.constant 5 : index
    %fc = arith.constant 5.0 : f32
    %A = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Af = linnea.fill(%fc, %A) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %B = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Bf = linnea.fill(%fc, %B) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %0 = linnea.equation {
      %1 = linnea.mul.high %Af, %Bf { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> -> !linnea.term
      linnea.yield %1 : !linnea.term
    }

    %fd = arith.constant 6.0 : f32
    %C = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
    %Cf = linnea.fill(%fd, %C) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    %D = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
    %Df = linnea.fill(%fd, %D) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    %1 = linnea.equation {
      %2 = linnea.mul.high %Cf, %Df, %0 { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.term -> !linnea.term
      %3 = linnea.mul.high %Cf, %Df, %2 { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.term -> !linnea.term
      linnea.yield %3 : !linnea.term
    }

    linnea.print %1 : !linnea.term
    return
  }
}

// -----

module {
  // CHECK: entry
  func @entry() {

    %c5 = arith.constant 5 : index
    %fc = arith.constant 5.0 : f32
    %A = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Af = linnea.fill(%fc, %A) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %B = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Bf = linnea.fill(%fc, %B) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %0 = linnea.equation {
      %1 = linnea.mul.high %Af, %Bf { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> -> !linnea.term
      linnea.yield %1 : !linnea.term
    }

    %fd = arith.constant 6.0 : f32
    %C = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
    %Cf = linnea.fill(%fd, %C) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    %D = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
    %Df = linnea.fill(%fd, %D) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    %1 = linnea.equation {
      %2 = linnea.mul.high %Cf, %Df, %0 { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.term -> !linnea.term
      %3 = linnea.mul.high %Cf, %Df, %2 { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.term -> !linnea.term
      linnea.yield %3 : !linnea.term
    }
    
    linnea.print %0 : !linnea.term
    linnea.print %1 : !linnea.term
    return
  }
}
