// RUN: standalone-opt %s | standalone-opt | FileCheck %s

func private @check_encoding(tensor<32x32xf64, #standalone.matrix<{encodingType = ["diagonal"]}>>)
