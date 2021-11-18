// RUN: standalone-opt --comprehensive-properties-propagation --convert-linnea-to-linalg %s | FileCheck %s

func @bar(%arg0: !linnea.matrix<["square"], [30, 35], f32>, 
          %arg1: !linnea.matrix<["square"], [35, 15], f32>,
          %arg2: !linnea.matrix<["square"], [15, 5], f32>,
          %arg3: !linnea.matrix<["square"], [5, 10], f32>,
          %arg4: !linnea.matrix<["square"], [10, 20], f32>,
          %arg5: !linnea.matrix<["square"], [20, 25], f32>) -> !linnea.term {

  %0 = linnea.equation {
    %6 = linnea.mul %arg0, %arg1, %arg2, %arg3, 
                    %arg4, %arg5 : !linnea.matrix<["square"], [30, 35], f32>, 
                                   !linnea.matrix<["square"], [35, 15], f32>, 
                                   !linnea.matrix<["square"], [15, 5], f32>, 
                                   !linnea.matrix<["square"], [5, 10], f32>, 
                                   !linnea.matrix<["square"], [10, 20], f32>, 
                                   !linnea.matrix<["square"], [20, 25], f32> -> !linnea.term 
      linnea.yield %6 : !linnea.term
  }
  return %0 : !linnea.term
}
