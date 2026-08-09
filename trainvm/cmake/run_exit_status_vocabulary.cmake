# Pins the CLI exit-status vocabulary declared in
# trainvm/include/trainvm/exit_status.hpp.
#
# The vocabulary is only useful to a caller if it is stable, and before this
# test nothing defended it: `trainvm_cli_qualify_evidence_*` pinned 0 and 3, and
# every other CLI test asserted only "zero". So the malformed code, the
# not-found code, and the usage code could all move without a red check, which
# is how the commands came to disagree in the first place -- `validate` answered
# 2 for a document that would not compile while `qualify-evidence` answered 1.
#
# Each code below is reached through more than one command where more than one
# command can reach it, because the claim being defended is that a code means
# one thing everywhere, not that some single command happens to return it.
#
# The pair that matters most is 2 against 3: malformed input must never collide
# with a negative verdict, and neither may collide with 1, an uncaught
# exception. A caller that cannot tell "your document is broken" from "trainvm
# is broken" has to treat both as fatal, which discards the distinction the
# qualification gate exists to report.

if(NOT DEFINED TRAINVM_BINARY)
  message(FATAL_ERROR "TRAINVM_BINARY is required")
endif()
if(NOT DEFINED TRAINVM_SOURCE_ROOT)
  message(FATAL_ERROR "TRAINVM_SOURCE_ROOT is required")
endif()
if(NOT DEFINED TRAINVM_SCRATCH_DIR)
  message(FATAL_ERROR "TRAINVM_SCRATCH_DIR is required")
endif()

set(examples "${TRAINVM_SOURCE_ROOT}/docs/experiment-vm/examples")

# Fixtures are written rather than committed. Each is the smallest document that
# reaches the failure being pinned, and committing four one-line files whose
# only property is "is rejected" would invite someone to make them realistic.
file(REMOVE_RECURSE "${TRAINVM_SCRATCH_DIR}")
file(MAKE_DIRECTORY "${TRAINVM_SCRATCH_DIR}")
set(not_json "${TRAINVM_SCRATCH_DIR}/not-json.txt")
set(uncompilable "${TRAINVM_SCRATCH_DIR}/uncompilable-experiment.json")
set(off_schema "${TRAINVM_SCRATCH_DIR}/off-schema-evidence.json")
set(journal "${TRAINVM_SCRATCH_DIR}/exit-status.db")
file(WRITE "${not_json}" "this is not a JSON document")
file(WRITE "${uncompilable}" "{\"api_version\": \"trainvm.rwkv-lab/v1alpha1\"}")
file(WRITE "${off_schema}"
  "{\"api_version\": \"trainvm.cache-qualification-evidence/v1\",\
 \"unexpected_field\": 1}")

set(failures "")

# Runs the binary and compares its exit status against the expected code.
# INPUT is an optional file redirected to stdin; add_test cannot redirect one.
function(expect_status label expected)
  cmake_parse_arguments(arg "" "INPUT" "ARGUMENTS" ${ARGN})
  if(DEFINED arg_INPUT)
    execute_process(
      COMMAND "${TRAINVM_BINARY}" ${arg_ARGUMENTS}
      INPUT_FILE "${arg_INPUT}"
      OUTPUT_VARIABLE output
      ERROR_VARIABLE errors
      RESULT_VARIABLE status
    )
  else()
    execute_process(
      COMMAND "${TRAINVM_BINARY}" ${arg_ARGUMENTS}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE errors
      RESULT_VARIABLE status
    )
  endif()
  if(status EQUAL expected)
    message("ok   ${label} -- exit ${expected}")
  else()
    # Every case is reported before the script fails, so one run names every
    # code that moved rather than only the first.
    string(APPEND failures
      "\n  ${label}: expected exit ${expected}, got ${status}"
      "\n    stderr: ${errors}")
    set(failures "${failures}" PARENT_SCOPE)
    message("FAIL ${label} -- expected ${expected}, got ${status}")
  endif()
endfunction()

# 64 -- the argument vector itself is wrong. Two shapes: an unknown subcommand,
# and a known subcommand at the wrong arity.
expect_status("unknown subcommand is a usage error" 64
  ARGUMENTS not-a-subcommand)
expect_status("recipe without an action is a usage error" 64
  ARGUMENTS recipe)

# 2 -- malformed input, through four commands. `validate` and `compile` read a
# document that parses but does not compile; `qualify-evidence` reads evidence
# that is not JSON at all, then evidence that is JSON but off its declared
# schema. That last pair is what this card changed: both answered 1 before,
# which is also what an uncaught exception answers.
expect_status("validate rejects a document that does not compile" 2
  ARGUMENTS validate "${uncompilable}")
expect_status("compile rejects a document that does not compile" 2
  ARGUMENTS compile INPUT "${uncompilable}")
expect_status("qualify-evidence rejects evidence that is not JSON" 2
  ARGUMENTS qualify-evidence INPUT "${not_json}")
expect_status("qualify-evidence rejects evidence off its schema" 2
  ARGUMENTS qualify-evidence INPUT "${off_schema}")

# 2 again -- a catalog that does not match the tree it is pinned against, and a
# catalog that cannot be read. Both reached the top-level catch and exited 1
# before, making "the catalog is wrong" indistinguishable from "trainvm
# crashed". The repository root below is deliberately the wrong directory, which
# is the cheapest way to make a valid catalog not match without editing it.
expect_status("validate-catalog rejects a catalog that does not match" 2
  ARGUMENTS validate-catalog
    "${TRAINVM_SOURCE_ROOT}/docs/experiment-vm/compatibility-workflows.v1.json"
    "${examples}")
expect_status("print-catalog-digests rejects an unreadable catalog" 2
  ARGUMENTS print-catalog-digests "${not_json}" "${TRAINVM_SOURCE_ROOT}")

# 3 -- a well-formed document whose verdict is no, and 0 for the affirmative
# direction of the same command, so a build in which everything exits non-zero
# cannot pass this test.
expect_status("qualify-evidence reports a rejection as a verdict" 3
  ARGUMENTS qualify-evidence
  INPUT "${examples}/qualification-evidence.rejected.v1.json")
expect_status("qualify-evidence reports a qualification" 0
  ARGUMENTS qualify-evidence
  INPUT "${examples}/qualification-evidence.qualified.v1.json")

# 4 -- the document was read and the thing it named does not exist. `journal
# show` against a freshly initialized journal is the only command that reaches
# this code today.
expect_status("journal init succeeds" 0
  ARGUMENTS journal init "${journal}")
expect_status("journal show reports an absent run as not found" 4
  ARGUMENTS journal show "${journal}" no-such-run)

if(NOT failures STREQUAL "")
  message(FATAL_ERROR "exit-status vocabulary violations:${failures}")
endif()
