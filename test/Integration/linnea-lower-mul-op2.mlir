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

    %1 = linnea.equation {
      %2 = linnea.mul.high %Af, %Bf, %0 { semirings = "real-arith" }:
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.term -> !linnea.term
      linnea.yield %2 : !linnea.term
    }
 
    //
    // CHECK:       ( ( 25, 0, 0, 0, 0 ),
    // CHECK-SAME:    ( 50, 25, 0, 0, 0 ),
    // CHECK-SAME:    ( 75, 50, 25, 0, 0 ),
    // CHECK-SAME:    ( 100, 75, 50, 25, 0 ),
    // CHECK-SAME:    ( 125, 100, 75, 50, 25 ) )
    linnea.print %0 : !linnea.term

    // CHECK:       ( ( 625, 0, 0, 0, 0 ),
    // CHECK-SAME:    ( 2500, 625, 0, 0, 0 ),
    // CHECK-SAME:    ( 6250, 2500, 625, 0, 0 ),
    // CHECK-SAME:    ( 12500, 6250, 2500, 625, 0 ),
    // CHECK-SAME:    ( 21875, 12500, 6250, 2500, 625 ) )
    linnea.print %1 : !linnea.term
    linnea.dealloc %A : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    linnea.dealloc %B : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    return
  }
}

