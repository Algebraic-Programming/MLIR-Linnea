#include "Standalone/LinneaDialect.h"
#include "Standalone/LinneaPasses.h"
#include "Standalone/LinneaTypes.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// see: ./lib/Dialect/TorchConversion/Transforms/BackendTypeConversion.cpp

using namespace mlir;
using namespace mlir::linnea;

#define GEN_PASS_CLASSES
#include "Standalone/LinneaPasses.h.inc"

namespace {
static void setupTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter) {
  typeConverter.addConversion([](MatrixType type) -> RankedTensorType {
    return RankedTensorType::get(type.getDims(), type.getElementType(),
                                 type.getProperty());
  });
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
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<CallOp>(
        [&](CallOp op) { return typeConverter.isLegal(op); });

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
