set_target_properties(StandaloneUnitTests PROPERTIES FOLDER "Standalone Tests")

function(add_linnea_unittest name)
  add_unittest(StandaloneUnitTests ${name} ${ARGN})
endfunction()

add_linnea_unittest(LinneaUnitTests dummy.cpp test-chain.cpp)

target_link_libraries(LinneaUnitTests
  PRIVATE
  MLIRLinnea
  )
