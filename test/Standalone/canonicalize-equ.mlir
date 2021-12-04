// RUN: standalone-opt --comprehensive-properties-propagation --convert-linnea-to-linalg %s | FileCheck %s
// XFAIL: *
func @bar(%arg0: !linnea.matrix<["fullrank"], [30, 35], f32>, 
          %arg1: !linnea.matrix<["fullrank"], [35, 15], f32>,
          %arg2: !linnea.matrix<["fullrank"], [15, 5], f32>,
          %arg3: !linnea.matrix<["fullrank"], [5, 10], f32>,
          %arg4: !linnea.matrix<["fullrank"], [10, 20], f32>,
          %arg5: !linnea.matrix<["fullrank"], [20, 25], f32>) -> !linnea.term {

  %0 = linnea.equation {
    %6 = linnea.mul %arg0, %arg1, %arg2, %arg3, 
                    %arg4, %arg5 : !linnea.matrix<["fullrank"], [30, 35], f32>, 
                                   !linnea.matrix<["fullrank"], [35, 15], f32>, 
                                   !linnea.matrix<["fullrank"], [15, 5], f32>, 
                                   !linnea.matrix<["fullrank"], [5, 10], f32>, 
                                   !linnea.matrix<["fullrank"], [10, 20], f32>, 
                                   !linnea.matrix<["fullrank"], [20, 25], f32> -> !linnea.term 
      linnea.yield %6 : !linnea.term
  }
  return %0 : !linnea.term
}
