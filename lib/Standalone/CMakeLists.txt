add_mlir_dialect_library(MLIRStandalone
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandaloneTypes.cpp
        StandaloneAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen
        MLIRStandaloneTypeDefIncGen
        MLIRStandaloneAttrDefIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
