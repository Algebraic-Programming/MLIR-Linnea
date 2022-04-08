// RUN: standalone-opt %s --linnea-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
module {
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
      %2 = linnea.mul.high %Cf, %Df { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32> -> !linnea.term
      %3 = linnea.add.high %2, %0 { semirings = "real-arith" }:
        !linnea.term, !linnea.term -> !linnea.term
      linnea.yield %3 : !linnea.term
    }

    //
    // CHECK:     ( ( 25, 0, 0, 0, 0 ),
    // CHECK-SAME:  ( 50, 25, 0, 0, 0 ),
    // CHECK-SAME:  ( 75, 50, 25, 0, 0 ),
    // CHECK-SAME:  ( 100, 75, 50, 25, 0 ),
    // CHECK-SAME:  ( 125, 100, 75, 50, 25 ) )
    linnea.print %0 : !linnea.term
    //
    // CHECK:     ( ( 61, 72, 108, 144, 180 ),
    // CHECK-SAME:  ( 50, 61, 72, 108, 144 ),
    // CHECK-SAME:  ( 75, 50, 61, 72, 108 ),
    // CHECK-SAME:  ( 100, 75, 50, 61, 72 ),
    // CHECK-SAME:  ( 125, 100, 75, 50, 61 ) )
    linnea.print %1 : !linnea.term
    linnea.dealloc %A : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    linnea.dealloc %B : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    linnea.dealloc %C : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32> 
    linnea.dealloc %D : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    return 
  }
}
