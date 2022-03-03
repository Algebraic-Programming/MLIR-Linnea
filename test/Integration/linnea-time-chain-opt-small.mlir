// RUN: ulimit -s 16000
// 
// RUN: standalone-opt %s --linnea-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
module {
  func @entry() {
  
    // A1 (800 x 1100)
    %fA1 = arith.constant 1 : i32
    %c800 = arith.constant 800 : index
    %c1100 = arith.constant 1100 : index
    %A1 = linnea.init [%c800, %c1100] : !linnea.matrix<#linnea.property<["general"]>, [800, 1100], i32>
    %Af1 = linnea.fill(%fA1, %A1) : i32, !linnea.matrix<#linnea.property<["general"]>, [800, 1100], i32>

    // A2 (1100 x 900)
    %fA2 = arith.constant 2 : i32
    %c900 = arith.constant 900 : index
    %A2 = linnea.init [%c1100, %c900] : !linnea.matrix<#linnea.property<["general"]>, [1100, 900], i32>
    %Af2 = linnea.fill(%fA2, %A2) : i32, !linnea.matrix<#linnea.property<["general"]>, [1100, 900], i32>

    // A3 (900 x 1200)
    %fA3 = arith.constant 3 : i32
    %c1200 = arith.constant 1200 : index
    %A3 = linnea.init [%c900, %c1200] : !linnea.matrix<#linnea.property<["general"]>, [900, 1200], i32>
    %Af3 = linnea.fill(%fA3, %A3) : i32, !linnea.matrix<#linnea.property<["general"]>, [900, 1200], i32>
    
    // A4 (1200 x 100)
    %fA4 = arith.constant 4 : i32
    %c100 = arith.constant 100 : index
    %A4 = linnea.init [%c1200, %c100] : !linnea.matrix<#linnea.property<["general"]>, [1200, 100], i32>
    %Af4 = linnea.fill(%fA4, %A4) : i32, !linnea.matrix<#linnea.property<["general"]>, [1200, 100], i32>
     
    // Too much stack allocation. To run use ulimit -s 16000
    %t_start_matmul = call @rtclock() : () -> f64 
    %0 = linnea.equation {
      %1 = linnea.mul.high %Af1, %Af2, %Af3, %Af4 :
        !linnea.matrix<#linnea.property<["general"]>, [800, 1100], i32>,
        !linnea.matrix<#linnea.property<["general"]>, [1100, 900], i32>,
        !linnea.matrix<#linnea.property<["general"]>, [900, 1200], i32>,
        !linnea.matrix<#linnea.property<["general"]>, [1200, 100], i32> -> !linnea.term
      linnea.yield %1 : !linnea.term
    }
    %t_end_matmul = call @rtclock() : () -> f64
    %tmatmul = arith.subf %t_end_matmul, %t_start_matmul: f64
    vector.print %tmatmul : f64
   
    // CHECK: ( ( TODO, 
    linnea.print %0 : !linnea.term  
    return 
  }

  func private @rtclock() -> f64
}
