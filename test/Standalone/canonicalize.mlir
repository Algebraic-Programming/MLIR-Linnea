// RUN: standalone-opt %s --split-input-file --canonicalize | FileCheck %s
// XFAILS: *

#GE = #linnea.matrix<{p = ["general"]}>
// CHECK-LABEL: func @bar(%{{.*}}: tensor<2x2xf64, #linnea.matrix<{p = ["general"]}>>)
func @bar(%arg0 : tensor<2x2xf64, #GE>) {
  // CHECK: return
  %0 = linnea.inverse %arg0 : tensor<2x2xf64, #GE> -> tensor<2x2xf64, #GE>
  %1 = linnea.inverse %0 : tensor<2x2xf64, #GE> -> tensor<2x2xf64, #GE>
  return
}

// -----

// CHECK-LABEL: func @bar(%{{.*}}: tensor<2x2xf64>)
func @bar(%arg0 : tensor<2x2xf64>) {
  // CHECK: return
  %0 = linnea.inverse %arg0 : tensor<2x2xf64> -> tensor<2x2xf64>
  %1 = linnea.inverse %0 : tensor<2x2xf64> -> tensor<2x2xf64>
  return
}

// -----

#S = #linnea.matrix<{p = ["spd"]}>
#F = #linnea.matrix<{p = ["fullrank", "square"]}>
func @bar(%A : tensor<2x2xf64, #F>, %S : tensor<2x2xf64, #S>) -> tensor<2x2xf64, #S> {
  %0 = linnea.inverse %S : tensor<2x2xf64, #S> -> tensor<2x2xf64, #S>
  %1 = linnea.transpose %A : tensor<2x2xf64, #F> -> tensor<2x2xf64, #F>
  %2 = linnea.mul %1, %0, %A : tensor<2x2xf64, #F>, tensor<2x2xf64, #S>, tensor<2x2xf64, #F> -> tensor<2x2xf64, #S>
  return %2 : tensor<2x2xf64, #S>
}
