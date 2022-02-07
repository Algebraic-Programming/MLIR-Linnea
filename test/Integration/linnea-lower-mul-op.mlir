// RUN: standalone-opt %s \
// RUN: --comprehensive-properties-propagation --linnea-func-type-conversion \
// RUN: --convert-linnea-to-linalg --convert-linnea-to-loops \ 
// RUN: --linnea-finalize-func-type-conversion --canonicalize --linalg-bufferize \ 
// RUN: --func-bufferize --arith-bufferize --tensor-bufferize \
// RUN: --finalizing-bufferize --convert-linalg-to-loops --convert-vector-to-scf \
// RUN: --convert-scf-to-cf --convert-arith-to-llvm --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

module {
  func @entry() {
  
    %c5 = arith.constant 5 : index

    // Materialize and fill a linnea matrix.
    %fc = arith.constant 5.0 : f32
    %A = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Af = linnea.fill(%fc, %A) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    // Materialize and fill a linnea matrix.
    %B = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %Bf = linnea.fill(%fc, %B) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    %0 = linnea.equation {
      %1 = linnea.mul.high %Af, %Bf :
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>,
        !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> -> !linnea.term
      linnea.yield %1 : !linnea.term
    }
    
    // this linnea.term gives a lot of problem. It is nice
    // to have the compiler keeping track of the result of
    // a linnea equation but how can we propagate the result
    // of a linnea equation to others?
    //
    // CHECK:       ( ( 25, 0, 0, 0, 0 ),
    // CHECK-SAME:    ( 50, 25, 0, 0, 0 ),
    // CHECK-SAME:    ( 75, 50, 25, 0, 0 ),
    // CHECK-SAME:    ( 100, 75, 50, 25, 0 ),
    // CHECK-SAME:    ( 125, 100, 75, 50, 25 ) )
    linnea.print %0 : !linnea.term 
    linnea.print %Af : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    linnea.print %Bf : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> 
    
    return 
  }
}
