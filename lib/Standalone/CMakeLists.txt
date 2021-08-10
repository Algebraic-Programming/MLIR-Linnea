add_mlir_dialect_library(MLIRLinnea
        LinneaDialect.cpp
        LinneaOps.cpp
        LinneaTypes.cpp
        LinneaAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRLinneaOpsIncGen
        MLIRLinneaTypeDefIncGen
        MLIRLinneaAttrDefIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
