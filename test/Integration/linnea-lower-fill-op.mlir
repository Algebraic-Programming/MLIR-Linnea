// RUN: standalone-opt %s --linnea-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

module {
  func @entry() {
    // materialize and fill a linnea matrix.
    %d1 = arith.constant -1.0 : f32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %fc = arith.constant 23.0 : f32
    %t = linnea.alloc [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    %f = linnea.fill(%fc, %t) : f32, !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
    
    // convert to memref, and print.
    %l = linnea.to_builtin_tensor %f :
      !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32> -> 
      tensor<5x5xf32, #linnea.property<["lowerTri"]>>
    %c = linnea.cast_to_builtin_tensor %l : 
      tensor<5x5xf32, #linnea.property<["lowerTri"]>> -> tensor<5x5xf32>
    %m = bufferization.to_memref %c : memref<5x5xf32>
    %v1 = vector.transfer_read %m[%c0, %c0], %d1 : memref<5x5xf32>, vector<5x5xf32>
    //
    // CHECK:     ( ( 23, 0, 0, 0, 0 ),
    // CHECK-SAME:  ( 23, 23, 0, 0, 0 ),
    // CHECK-SAME:  ( 23, 23, 23, 0, 0 ),
    // CHECK-SAME:  ( 23, 23, 23, 23, 0 ),
    // CHECK-SAME:  ( 23, 23, 23, 23, 23 ) ) 
    vector.print %v1 : vector<5x5xf32>
    linnea.dealloc %t : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>

    return 
  }
}
