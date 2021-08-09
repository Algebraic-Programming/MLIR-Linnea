add_mlir_dialect(StandaloneOps standalone)
add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS StandaloneTypeBase.td)
mlir_tablegen(StandaloneTypeBase.h.inc -gen-typedef-decls -typedefs-dialect=standalone)
mlir_tablegen(StandaloneTypeBase.cpp.inc -gen-typedef-defs -typedefs-dialect=standalone)
add_public_tablegen_target(MLIRStandaloneTypeDefIncGen)

set(LLVM_TARGET_DEFINITIONS StandaloneAttrBase.td)
mlir_tablegen(StandaloneAttrBase.h.inc -gen-attrdef-decls)
mlir_tablegen(StandaloneAttrBase.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRStandaloneAttrDefIncGen)
