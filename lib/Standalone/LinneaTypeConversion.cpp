//===- LinneaTypeConversion.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaPasses.h"
#include "Standalone/LinneaTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// see: ./lib/Dialect/TorchConversion/Transforms/BackendTypeConversion.cpp

using namespace mlir;
using namespace mlir::func;
using namespace mlir::linnea;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

namespace {
static void setupTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter) {
  target.addLegalOp<ToBuiltinTensorOp>();
  typeConverter.addConversion([](MatrixType type) -> RankedTensorType {
    return RankedTensorType::get(type.getDims(), type.getElementType(),
                                 type.getProperty());
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, TensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<MatrixType>());
    return builder.create<ToBuiltinTensorOp>(loc, type, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder, Type type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<RankedTensorType>());
    return builder.create<FromBuiltinTensorOp>(loc, type, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

struct LinneaFuncTypeConversionPass
    : public LinneaFuncTypeConversionBase<LinneaFuncTypeConversionPass> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    typeConverter.addConversion([](Type type) { return type; });
    setupTypeConversion(target, typeConverter);

    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::linnea::createLinneaFuncTypeConversion() {
  return std::make_unique<LinneaFuncTypeConversionPass>();
}
