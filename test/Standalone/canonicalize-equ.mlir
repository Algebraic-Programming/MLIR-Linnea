// RUN: standalone-opt --split-input-file %s | FileCheck %s

  func @bar(%arg0: !linnea.matrix<["fullrank", "square"], [2, 2], f32>, %arg1: !linnea.matrix<["square"],[2, 2], f32>) -> !linnea.term {
    %0 = linnea.equ %arg0, %arg1 : !linnea.matrix<["fullrank", "square"], [2, 2], f32>, !linnea.matrix<["square"],[2, 2], f32> -> !linnea.term  {
    ^bb0(%0: !linnea.matrix<["fullrank", "square"], [2, 2], f32>, %1 : !linnea.matrix<["square"],[2, 2], f32>):  // no predecessors
      %2 = linnea.transpose %0 : !linnea.matrix<["fullrank", "square"], [2, 2], f32> -> !linnea.term
      %3 = linnea.mul %2, %0, %1 : !linnea.term, !linnea.matrix<["fullrank", "square"], [2, 2], f32>, !linnea.matrix<["square"], [2, 2], f32> -> !linnea.term
      linnea.yield %3 : !linnea.term
    }
    return %0 : !linnea.term
  }
