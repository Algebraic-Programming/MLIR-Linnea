add_mlir_public_c_api_library(StandaloneCAPI
  LinneaDialect.cpp
  LinneaPasses.cpp

  LINK_LIBS PUBLIC
  MLIRLinnea
) 
