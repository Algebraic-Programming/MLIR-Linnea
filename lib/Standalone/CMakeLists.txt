add_mlir_dialect_library(MLIRLinnea
        LinneaDialect.cpp
        LinneaOps.cpp
        LinneaTypes.cpp
        LinneaAttributes.cpp
        LinneaPropertyPropagation.cpp
        LinneaExpr.cpp
        LinneaProperties.cpp
        LinneaUtils.cpp
        LinneaUtilsExpr.cpp
        LinneaConvertToLoops.cpp
        LinneaConvertToLinalg.cpp
        LinneaCompilerPipeline.cpp
        LinneaExtraCopyRemoval.cpp
        LinneaTypeConversion.cpp
        LinneaFinalizeTypeConversion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRLinneaOpsIncGen
        MLIRLinneaTypeDefIncGen
        MLIRLinneaAttrDefIncGen
        MLIRLinneaPassIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
