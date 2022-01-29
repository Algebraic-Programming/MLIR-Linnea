// RUN: standalone-opt %s \
// RUN: --comprehensive-properties-propagation --linnea-func-type-conversion \
// RUN: --convert-linnea-to-linalg --convert-linnea-to-loops \ 
// RUN: --linnea-finalize-func-type-conversion --linalg-bufferize \ 
// RUN: --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN: --std-bufferize --finalizing-bufferize --convert-vector-to-scf \
// RUN: --convert-scf-to-std --convert-arith-to-llvm --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
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
    %t = linnea.init [%c5, %c5] : !linnea.matrix<#linnea.property<["lowerTri"]>, [5, 5], f32>
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
    // CHECK-SAME:  ( 23, 23, 23, 23, 23 ) ) 
    vector.print %v1 : vector<5x5xf32>
    memref.dealloc %m : memref<5x5xf32>
    return 
  }
}
