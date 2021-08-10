add_mlir_dialect(LinneaOps linnea)
add_mlir_doc(LinneaDialect LinneaDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(LinneaOps LinneaOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS LinneaTypeBase.td)
mlir_tablegen(LinneaTypeBase.h.inc -gen-typedef-decls -typedefs-dialect=linnea)
mlir_tablegen(LinneaTypeBase.cpp.inc -gen-typedef-defs -typedefs-dialect=linnea)
add_public_tablegen_target(MLIRLinneaTypeDefIncGen)

set(LLVM_TARGET_DEFINITIONS LinneaAttrBase.td)
mlir_tablegen(LinneaAttrBase.h.inc -gen-attrdef-decls -typedefs-dialect=linnea)
mlir_tablegen(LinneaAttrBase.cpp.inc -gen-attrdef-defs -typedefs-dialect=linnea)
add_public_tablegen_target(MLIRLinneaAttrDefIncGen)
