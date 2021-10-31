add_mlir_dialect_library(MLIRLinnea
        LinneaDialect.cpp
        LinneaOps.cpp
        LinneaTypes.cpp
        LinneaAttributes.cpp
        LinneaPasses.cpp
        LinneaExpr.cpp
        LinneaProperties.cpp
        LinneaUtils.cpp

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
