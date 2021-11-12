// RUN: standalone-opt --split-input-file --comprehensive-properties-propagation --convert-linnea-to-linalg %s | FileCheck %s

// CHECK:  func @bar(%arg0: tensor<30x35xf32>, %arg1: tensor<35x15xf32>, %arg2: tensor<15x5xf32>, %arg3: tensor<5x10xf32>, %arg4: tensor<10x20xf32>, %arg5: tensor<20x25xf32>) -> tensor<30x25xf32> {
// CHECK-NEXT:    %0 = linalg.init_tensor [30, 15] : tensor<30x15xf32>
// CHECK-NEXT:    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<30x35xf32>, tensor<35x15xf32>) outs(%0 : tensor<30x15xf32>) -> tensor<30x15xf32>
// CHECK-NEXT:    %2 = linalg.init_tensor [30, 5] : tensor<30x5xf32>
// CHECK-NEXT:    %3 = linalg.matmul ins(%1, %arg2 : tensor<30x15xf32>, tensor<15x5xf32>) outs(%2 : tensor<30x5xf32>) -> tensor<30x5xf32>
// CHECK-NEXT:    %4 = linalg.init_tensor [30, 10] : tensor<30x10xf32>
// CHECK-NEXT:    %5 = linalg.matmul ins(%3, %arg3 : tensor<30x5xf32>, tensor<5x10xf32>) outs(%4 : tensor<30x10xf32>) -> tensor<30x10xf32>
// CHECK-NEXT:    %6 = linalg.init_tensor [30, 20] : tensor<30x20xf32>
// CHECK-NEXT:    %7 = linalg.matmul ins(%5, %arg4 : tensor<30x10xf32>, tensor<10x20xf32>) outs(%6 : tensor<30x20xf32>) -> tensor<30x20xf32>
// CHECK-NEXT:    %8 = linalg.init_tensor [30, 25] : tensor<30x25xf32>
// CHECK-NEXT:    %9 = linalg.matmul ins(%7, %arg5 : tensor<30x20xf32>, tensor<20x25xf32>) outs(%8 : tensor<30x25xf32>) -> tensor<30x25xf32>
// CHECK-NEXT:    return %9 : tensor<30x25xf32>
// CHECK-NEXT:  }

func @bar(%arg0: !linnea.matrix<["square"], [30, 35], f32>, 
          %arg1: !linnea.matrix<["square"], [35, 15], f32>,
          %arg2: !linnea.matrix<["square"], [15, 5], f32>,
          %arg3: !linnea.matrix<["square"], [5, 10], f32>,
          %arg4: !linnea.matrix<["square"], [10, 20], f32>,
          %arg5: !linnea.matrix<["square"], [20, 25], f32>) -> !linnea.term {

  %0 = linnea.equ %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !linnea.matrix<["square"], [30, 35], f32>, 
                                                             !linnea.matrix<["square"], [35, 15], f32>,
                                                             !linnea.matrix<["square"], [15, 5], f32>,
                                                             !linnea.matrix<["square"], [5, 10], f32>,
                                                             !linnea.matrix<["square"], [10, 20], f32>,
                                                             !linnea.matrix<["square"], [20, 25], f32> -> !linnea.term  {
    ^bb0(%0: !linnea.matrix<["square"], [30, 35], f32>,
         %1: !linnea.matrix<["square"], [35, 15], f32>,
         %2: !linnea.matrix<["square"], [15, 5], f32>,
         %3: !linnea.matrix<["square"], [5, 10], f32>,
         %4: !linnea.matrix<["square"], [10, 20], f32>,
         %5: !linnea.matrix<["square"], [20, 25], f32>):
      %6 = linnea.mul %0, %1, %2, %3, %4, %5 : !linnea.matrix<["square"], [30, 35], f32>, 
                                               !linnea.matrix<["square"], [35, 15], f32>, 
                                               !linnea.matrix<["square"], [15, 5], f32>, 
                                               !linnea.matrix<["square"], [5, 10], f32>, 
                                               !linnea.matrix<["square"], [10, 20], f32>, 
                                               !linnea.matrix<["square"], [20, 25], f32> -> !linnea.term 
      linnea.yield %6 : !linnea.term
  }
  return %0 : !linnea.term
}
