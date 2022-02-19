if (MLIR_ENABLE_BINDINGS_PYTHON AND ENABLE_PYLINNEA)
  message(STATUS "Enabling pylinnea")
  add_subdirectory(pylinnea)
endif()
