// RUN: standalone-opt --split-input-file %s | standalone-opt | FileCheck %s

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.matrix<#linnea.property<["general"]>, [32, 32], f32>)
func private @check_type(%arg0: !linnea.matrix<#linnea.property<["general"]>,[32, 32], f32>)

// -----

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.matrix<#linnea.property<["lowerTri"]>, [3, 3], f32>)
func private @check_type(%arg0: !linnea.matrix<#linnea.property<["lowerTri"]>,[3, 3], f32>)

// -----

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.term
func private @check_type(%arg0: !linnea.term)

// -----

// CHECK: func private @check_type(
// CHECK-SAME: !linnea.identity<[32, 32], f32>)
func private @check_type(%arg0 : !linnea.identity<[32, 32], f32>)
