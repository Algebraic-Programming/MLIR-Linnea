// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar(%arg0: !standalone.matrix<tensor<32x32xf64>, 0>) {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
        %res = standalone.foo %0 : i32
        return
    }
}

//%0 = matrix<32x32xf32>
