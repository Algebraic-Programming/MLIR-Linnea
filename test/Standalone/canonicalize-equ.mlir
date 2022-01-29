// RUN: standalone-opt --split-input-file --comprehensive-properties-propagation --convert-linnea-to-linalg %s | FileCheck %s
// XFAIL: *
func @bar(%arg0: !linnea.matrix<#linnea.property<["fullrank"]>, [30, 35], f32>, 
          %arg1: !linnea.matrix<#linnea.property<["fullrank"]>, [35, 15], f32>,
          %arg2: !linnea.matrix<#linnea.property<["fullrank"]>, [15, 5], f32>,
          %arg3: !linnea.matrix<#linnea.property<["fullrank"]>, [5, 10], f32>,
          %arg4: !linnea.matrix<#linnea.property<["fullrank"]>, [10, 20], f32>,
          %arg5: !linnea.matrix<#linnea.property<["fullrank"]>, [20, 25], f32>) -> !linnea.term {

  %0 = linnea.equation {
    %6 = linnea.mul.high %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : 
                        !linnea.matrix<#linnea.property<["fullrank"]>, [30, 35], f32>, 
                        !linnea.matrix<#linnea.property<["fullrank"]>, [35, 15], f32>, 
                        !linnea.matrix<#linnea.property<["fullrank"]>, [15, 5], f32>, 
                        !linnea.matrix<#linnea.property<["fullrank"]>, [5, 10], f32>, 
                        !linnea.matrix<#linnea.property<["fullrank"]>, [10, 20], f32>, 
                        !linnea.matrix<#linnea.property<["fullrank"]>, [20, 25], f32> -> !linnea.term 
      linnea.yield %6 : !linnea.term
  }
  return %0 : !linnea.term
}

// -----

func @foo(%arg0: !linnea.matrix<#linnea.property<["square", "lowerTri"]>, 
                                                 [30, 30], f32>,
          %arg1: !linnea.matrix<#linnea.property<["square", "lowerTri"]>, 
                                                 [30, 30], f32>) -> !linnea.term {
  %0 = linnea.equation {
    %1 = linnea.mul.high %arg0, %arg1 : 
        !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32>,
        !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32> -> !linnea.term
    linnea.yield %1 : !linnea.term
  }
  return %0 : !linnea.term
}

// -----

func @foo(%arg0: !linnea.matrix<#linnea.property<["square", "lowerTri"]>, 
                                                 [30, 30], f32>,
          %arg1: !linnea.matrix<#linnea.property<["square", "lowerTri"]>, 
                                                 [30, 30], f32>,
          %arg2: f32) -> !linnea.term {
  linnea.fill(%arg2, %arg0) : 
    f32, !linnea.matrix<#linnea.property<["square", "lowerTri"]>,[30, 30], f32> 
  linnea.fill(%arg2, %arg1) : 
    f32, !linnea.matrix<#linnea.property<["square", "lowerTri"]>,[30, 30], f32>
  %0 = linnea.equation {
    %1 = linnea.mul.high %arg0, %arg1 : 
        !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32>,
        !linnea.matrix<#linnea.property<["square", "lowerTri"]>, [30, 30], f32> -> !linnea.term
    linnea.yield %1 : !linnea.term
  }
  return %0 : !linnea.term
}
