# Drives `trainvm qualify-evidence` with stdin redirected, because add_test
# cannot redirect input directly. The verdict is the exit status, so this
# asserts the exact code rather than only scraping stdout: a gate that crashed
# and a gate that rejected must never look the same.
if(NOT DEFINED TRAINVM_BINARY)
  message(FATAL_ERROR "TRAINVM_BINARY is required")
endif()
set(evidence "$ENV{TRAINVM_EVIDENCE}")
if(NOT EXISTS "${evidence}")
  message(FATAL_ERROR "evidence fixture not found: ${evidence}")
endif()
execute_process(
  COMMAND "${TRAINVM_BINARY}" qualify-evidence
  INPUT_FILE "${evidence}"
  OUTPUT_VARIABLE output
  ERROR_VARIABLE errors
  RESULT_VARIABLE status
)
message("${output}${errors}")
if(DEFINED ENV{TRAINVM_EXPECT_REJECTED})
  if(NOT status EQUAL 3)
    message(FATAL_ERROR "expected the rejected verdict (exit 3), got ${status}")
  endif()
elseif(NOT status EQUAL 0)
  message(FATAL_ERROR "expected the qualified verdict (exit 0), got ${status}")
endif()
