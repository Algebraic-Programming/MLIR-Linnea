// RUN: standalone-opt --split-input-file %s | FileCheck %s

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
