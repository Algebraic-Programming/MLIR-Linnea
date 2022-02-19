find_package(
  Python3
  REQUIRED
  COMPONENTS Interpreter
)

execute_process(
  COMMAND pip show TatSu
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET
)

if (NOT ${EXIT_CODE} EQUAL 0)
  message(FATAL_ERROR
    "The \"TatSu\" Python3 package is not installed. \"
    Please install it using the following command: \"pip3 install TatSu\"."
    )
endif()
