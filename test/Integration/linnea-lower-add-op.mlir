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
    %A = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Af = linnea.fill(%fc, %A) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %B = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Bf = linnea.fill(%fc, %B) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %0 = linnea.equation {
      %1 = linnea.mul.high %Af, %Bf :
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> -> !linnea.term
      linnea.yield %1 : !linnea.term
    }

    %fd = arith.constant 6.0 : f32
    %C = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
    %Cf = linnea.fill(%fd, %C) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    %D = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>
    %Df = linnea.fill(%fd, %D) : f32, !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>

    %1 = linnea.equation {
      %2 = linnea.mul.high %Cf, %Df :
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["upperTri"]>, [5, 5], f32> -> !linnea.term
      linnea.yield %2 : !linnea.term
    }

    //
    // CHECK:     ( ( 25, 0, 0, 0, 0 ),
    // CHECK-SAME:  ( 50, 25, 0, 0, 0 ),
    // CHECK-SAME:  ( 75, 50, 25, 0, 0 ),
    // CHECK-SAME:  ( 100, 75, 50, 25, 0 ),
    // CHECK-SAME:  ( 125, 100, 75, 50, 25 ) )
    linnea.print %0 : !linnea.term
    //
    // CHECK:     ( ( 36, 72, 108, 144, 180 ),
    // CHECK-SAME:  ( 0, 36, 72, 108, 144 ),
    // CHECK-SAME:  ( 0, 0, 36, 72, 108 ),
    // CHECK-SAME:  ( 0, 0, 0, 36, 72 ),
    // CHECK-SAME:  ( 0, 0, 0, 0, 36 ) )
    linnea.print %1 : !linnea.term
    
    return 
  }
}
