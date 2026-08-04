cmake_minimum_required(VERSION 3.25)

project(trainvm VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "TrainVM intentionally requires GCC 16+ for C++26 reflection")
endif()
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
  message(FATAL_ERROR "TrainVM requires GCC 16+; found ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# C++26 makes std::optional a range, which lets nlohmann silently encode it as
# []/[value]. Every native translation unit must pass through TrainVM's explicit
# null/scalar serializer so a new protocol field cannot regain that ambiguity.
file(GLOB_RECURSE TRAINVM_NATIVE_JSON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)
set(TRAINVM_JSON_WRAPPER
  "${CMAKE_CURRENT_SOURCE_DIR}/include/trainvm/json.hpp")
foreach(TRAINVM_JSON_SOURCE IN LISTS TRAINVM_NATIVE_JSON_SOURCES)
  if(NOT TRAINVM_JSON_SOURCE STREQUAL TRAINVM_JSON_WRAPPER)
    file(STRINGS "${TRAINVM_JSON_SOURCE}" TRAINVM_DIRECT_JSON_INCLUDES
      REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"]nlohmann/json\\.hpp[>\"]")
    if(TRAINVM_DIRECT_JSON_INCLUDES)
      message(FATAL_ERROR
        "${TRAINVM_JSON_SOURCE} bypasses trainvm/json.hpp")
    endif()
  endif()
endforeach()

get_filename_component(TRAINVM_PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../proto" REALPATH)
set(TRAINVM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(TRAINVM_PROTO "${TRAINVM_PROTO_ROOT}/trainvm/v1/trainvm.proto")
set(TRAINVM_PROTO_SOURCES
  "${TRAINVM_GENERATED_DIR}/trainvm/v1/trainvm.pb.cc"
  "${TRAINVM_GENERATED_DIR}/trainvm/v1/trainvm.grpc.pb.cc"
)
set(TRAINVM_PROTO_HEADERS
  "${TRAINVM_GENERATED_DIR}/trainvm/v1/trainvm.pb.h"
  "${TRAINVM_GENERATED_DIR}/trainvm/v1/trainvm.grpc.pb.h"
)
add_custom_command(
  OUTPUT ${TRAINVM_PROTO_SOURCES} ${TRAINVM_PROTO_HEADERS}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${TRAINVM_GENERATED_DIR}"
  COMMAND protobuf::protoc
    --proto_path=${TRAINVM_PROTO_ROOT}
    --cpp_out=${TRAINVM_GENERATED_DIR}
    --grpc_out=${TRAINVM_GENERATED_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${TRAINVM_PROTO}
  DEPENDS "${TRAINVM_PROTO}"
  VERBATIM
)
add_library(trainvm_protocol ${TRAINVM_PROTO_SOURCES} ${TRAINVM_PROTO_HEADERS})
target_include_directories(trainvm_protocol SYSTEM PUBLIC "${TRAINVM_GENERATED_DIR}")
target_link_libraries(trainvm_protocol PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_library(trainvm_core
  src/adapter_invocation.cpp
  src/adapter_registry.cpp
  src/authority_time.cpp
  src/authority_document.cpp
  src/cache_artifact_authority.cpp
  src/cache_namespace.cpp
  src/cache_namespace_authority.cpp
  src/control.cpp
  src/controller.cpp
  src/document.cpp
  src/eval_examples_contract.cpp
  src/external_profiler_artifact.cpp
  src/fake_worker.cpp
  src/fsm.cpp
  src/host_ledger.cpp
  src/sqlite_filesystem_authority.cpp
  src/host_process_saga.cpp
  src/host_startup_audit.cpp
  src/hostd.cpp
  src/hostd_boot_provisioning.cpp
  src/hostd_client_bootstrap.cpp
  src/hostd_client_configuration.cpp
  src/hostd_daemon_configuration.cpp
  src/hostd_crash_qualification.cpp
  src/hostd_daemon_runtime.cpp
  src/hostd_gpu_authorization.cpp
  src/hostd_journal_fence_attestor.cpp
  src/hostd_journal_logical_fence.cpp
  src/hostd_ledger_singleton_token.cpp
  src/hostd_linux_cgroup_authority.cpp
  src/hostd_linux_device_kernel.cpp
  src/hostd_linux_device_policy.cpp
  src/hostd_linux_inventory_context_auditor.cpp
  src/hostd_linux_process_authority.cpp
  src/hostd_linux_process_policy.cpp
  src/hostd_linux_process_policy_kernel.cpp
  src/hostd_linux_process_recovery.cpp
  src/hostd_linux_service_identity.cpp
  src/hostd_linux_session_authority.cpp
  src/hostd_linux_stopped_launcher.cpp
  src/hostd_mutation_client.cpp
  src/hostd_mutation_claim_provider.cpp
  src/hostd_mutation_protocol.cpp
  src/hostd_process_client.cpp
  src/hostd_process_protocol.cpp
  src/hostd_resource_client.cpp
  src/hostd_restart_process_recovery.cpp
  src/hostd_session_challenge.cpp
  src/hostd_startup_auditor.cpp
  src/hostd_startup_controller.cpp
  src/hostd_terminal_release_recovery.cpp
  src/hostd_transport.cpp
  src/host_launch.cpp
  src/host_launch_registry.cpp
  src/host_resources.cpp
  src/input_content_authority.cpp
  src/journal.cpp
  src/lifecycle_admission.cpp
  src/journal_cache_lease_authority.cpp
  src/lease_renewal.cpp
  src/linux_cache_evidence.cpp
  src/linux_immutable_cache_store.cpp
  src/linux_nvidia_inventory.cpp
  src/reconciler.cpp
  src/resource_request_builder.cpp
  src/run_authoring.cpp
  src/rwkv_lab_worker_contract.cpp
  src/conversion_publication.cpp
  src/post_training_authority.cpp
  src/profiler_launch_profiles.cpp
  src/recipe_profile.cpp
  src/rwkv_scratch_profiles.cpp
  src/sqlite_authority_vfs.cpp
  src/training_component_registry.cpp
  src/training_preflight.cpp
  src/worker_bootstrap.cpp
)
target_include_directories(trainvm_core PUBLIC include)
target_compile_options(trainvm_core PUBLIC -freflection)
target_compile_options(trainvm_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm_core
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE OpenSSL::Crypto SQLite3::SQLite3 yaml-cpp::yaml-cpp ${CMAKE_DL_LIBS}
)

add_library(trainvm_compatibility_catalog
  src/compatibility_catalog.cpp
  src/source_disposition_catalog.cpp
)
target_include_directories(trainvm_compatibility_catalog PUBLIC include)
target_compile_options(trainvm_compatibility_catalog PUBLIC -freflection)
target_compile_options(trainvm_compatibility_catalog PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(trainvm_compatibility_catalog
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE OpenSSL::Crypto
)

add_library(trainvm_experiment_analysis
  src/experiment_analysis.cpp
  src/experiment_registry.cpp
)
target_include_directories(trainvm_experiment_analysis PUBLIC include)
target_compile_options(trainvm_experiment_analysis PUBLIC -freflection)
target_compile_options(trainvm_experiment_analysis PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(trainvm_experiment_analysis
  PRIVATE SQLite3::SQLite3 nlohmann_json::nlohmann_json
)

add_library(trainvm_training_schedules
  src/training_schedules.cpp
)
target_include_directories(trainvm_training_schedules PUBLIC include)
target_compile_options(trainvm_training_schedules PUBLIC -freflection)
target_compile_options(trainvm_training_schedules PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)

add_library(trainvm_service src/service.cpp src/run_authoring_cli.cpp)
target_include_directories(trainvm_service PUBLIC include)
target_compile_options(trainvm_service PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm_service PUBLIC trainvm_core trainvm_protocol)

add_executable(trainvm src/main.cpp)
target_compile_options(trainvm PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  # GCC 16 at -O3 misattributes nlohmann's shared_ptr output-adapter storage to
  # std::mutex after inlining through the reflection-enabled CLI translation
  # unit. Keep the diagnostic visible, but do not turn this system-header false
  # positive into a release-build failure; every library/test TU retains full
  # -Werror, including -Warray-bounds.
  $<$<CONFIG:Release>:-Wno-error=array-bounds>
)
target_link_libraries(trainvm PRIVATE
  trainvm_service trainvm_compatibility_catalog trainvm_experiment_analysis
  trainvm_training_schedules
)

add_executable(trainvm-hostd src/hostd_main.cpp)
add_executable(trainvm-hostd-crash-qualification
  src/hostd_crash_qualification_main.cpp
)
target_compile_options(trainvm-hostd-crash-qualification PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(trainvm-hostd-crash-qualification PRIVATE trainvm_core)
target_compile_options(trainvm-hostd PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(trainvm-hostd PRIVATE trainvm_core)

enable_testing()
find_program(TRAINVM_GO_EXECUTABLE go)
find_program(TRAINVM_PYTHON_EXECUTABLE python3)
add_test(NAME trainvm_hostd_validate_configuration
  COMMAND trainvm-hostd --validate-config
    "${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/hostd-daemon.v1.json"
)
add_executable(trainvm_tests tests/trainvm_tests.cpp)
target_compile_options(trainvm_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  -fno-access-control
)
target_compile_definitions(trainvm_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(trainvm_tests PRIVATE trainvm_service)
add_test(NAME trainvm_tests COMMAND trainvm_tests)
# RUN_SERIAL removed: the cause it mitigated is gone. The suite's WorkerControl
# stream reads no longer assume the next frame is the reply to whatever was just
# written (see read_matching in trainvm/tests/trainvm_tests.cpp), which is what
# made it lose races under contention. Keeping a mitigation for a removed cause
# costs wall-clock on every CI run -- this is the largest suite in the tree --
# and quietly hides a recurrence.
#
# If that assertion ever fails again, restore RUN_SERIAL as the immediate
# mitigation and audit the stream reads for a frame kind read_matching does not
# yet distinguish.
if(TRAINVM_GO_EXECUTABLE)
  # The live stack runs the real service, which acquires its journal under the
  # strict_filesystem grade. That grade requires the whole authority ancestry to
  # be unwritable outside the authority owner, so the harness cannot place its
  # journal beneath a world-writable /tmp. Give it a journal directory in the
  # build tree, mirroring a real deployment directory such as /var/lib/trainvm.
  # Only the journal moves: the Unix socket stays under the shorter default
  # temporary directory because sun_path is limited to 108 bytes.
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/e2e-authority")
  add_test(NAME trainvm_dashboard_live_e2e
    COMMAND ${CMAKE_COMMAND} -E env
      TRAINVM_E2E_BINARY=$<TARGET_FILE:trainvm>
      TRAINVM_E2E_AUTHORITY_DIR=${CMAKE_CURRENT_BINARY_DIR}/e2e-authority
      ${TRAINVM_GO_EXECUTABLE} test ./internal/server
        -run ^TestTrainVMLiveStackE2E$ -count=1
  )
  set_tests_properties(trainvm_dashboard_live_e2e PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../dashboard"
    TIMEOUT 45
  )
endif()
add_executable(adapter_invocation_tests tests/adapter_invocation_tests.cpp)
target_compile_options(adapter_invocation_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(adapter_invocation_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(adapter_invocation_tests PRIVATE trainvm_core)
add_test(NAME adapter_invocation_tests COMMAND adapter_invocation_tests)
add_executable(worker_bootstrap_tests tests/worker_bootstrap_tests.cpp)
target_compile_options(worker_bootstrap_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(worker_bootstrap_tests PRIVATE trainvm_core)
add_test(NAME worker_bootstrap_tests COMMAND worker_bootstrap_tests)
add_executable(eval_examples_contract_tests
  tests/eval_examples_contract_tests.cpp
)
target_compile_options(eval_examples_contract_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(eval_examples_contract_tests PRIVATE trainvm_core)
add_test(NAME eval_examples_contract_tests COMMAND eval_examples_contract_tests)
add_executable(training_component_registry_tests
  tests/training_component_registry_tests.cpp
)
target_compile_options(training_component_registry_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(training_component_registry_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(training_component_registry_tests PRIVATE trainvm_core)
add_test(NAME training_component_registry_tests
  COMMAND training_component_registry_tests
)
add_executable(recipe_profile_tests
  tests/recipe_profile_tests.cpp
)
target_compile_options(recipe_profile_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(recipe_profile_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(recipe_profile_tests PRIVATE trainvm_core)
add_test(NAME recipe_profile_tests COMMAND recipe_profile_tests)
add_executable(rwkv_lab_worker_contract_tests
  tests/rwkv_lab_worker_contract_tests.cpp
)
target_compile_options(rwkv_lab_worker_contract_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(rwkv_lab_worker_contract_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(rwkv_lab_worker_contract_tests PRIVATE trainvm_core)
add_test(NAME rwkv_lab_worker_contract_tests
  COMMAND rwkv_lab_worker_contract_tests
)
if(TRAINVM_PYTHON_EXECUTABLE)
  add_test(NAME rwkv_lab_worker_contract_python_parity
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../src
      ${TRAINVM_PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_rwkv_lab_worker_contract.py
      $<TARGET_FILE:trainvm>
  )
  # Deliberately NOT part of rwkv_lab_worker_artifact: that test builds a
  # sealed worker artifact and is excluded from the hosted native job, which
  # is how a stale adapter pin survived for hours. This one needs only the
  # binary, so it runs everywhere the suite runs.
  add_test(NAME rwkv_lab_runtime_requirements
    COMMAND ${TRAINVM_PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_rwkv_lab_runtime_requirements.py
      $<TARGET_FILE:trainvm>
  )
  add_test(NAME rwkv_lab_worker_artifact
    COMMAND ${TRAINVM_PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_rwkv_lab_worker_artifact.py
      ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build_trainvm_worker_artifact.py
      ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build_trainvm_runtime_closure.py
      ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/materialize_trainvm_worker_deployment.py
      $<TARGET_FILE:trainvm>
      ${CMAKE_CURRENT_SOURCE_DIR}/../src
  )
endif()
add_executable(authority_time_tests tests/authority_time_tests.cpp)
target_compile_options(authority_time_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(authority_time_tests PRIVATE trainvm_core)
add_test(NAME authority_time_tests COMMAND authority_time_tests)
add_executable(sqlite_authority_vfs_tests tests/sqlite_authority_vfs_tests.cpp)
target_compile_options(sqlite_authority_vfs_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(sqlite_authority_vfs_tests PRIVATE trainvm_core)
add_test(NAME sqlite_authority_vfs_tests COMMAND sqlite_authority_vfs_tests)
add_executable(cache_namespace_tests tests/cache_namespace_tests.cpp)
target_compile_options(cache_namespace_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(cache_namespace_tests PRIVATE trainvm_core)
add_test(NAME cache_namespace_tests COMMAND cache_namespace_tests)
add_executable(cache_namespace_authority_tests
  tests/cache_namespace_authority_tests.cpp)
target_compile_options(cache_namespace_authority_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(cache_namespace_authority_tests PRIVATE trainvm_core)
add_test(NAME cache_namespace_authority_tests
  COMMAND cache_namespace_authority_tests)
add_executable(compatibility_catalog_tests tests/compatibility_catalog_tests.cpp)
target_compile_options(compatibility_catalog_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(compatibility_catalog_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(compatibility_catalog_tests PRIVATE trainvm_compatibility_catalog)
add_test(NAME compatibility_catalog_tests COMMAND compatibility_catalog_tests)
add_executable(source_disposition_catalog_tests
  tests/source_disposition_catalog_tests.cpp
)
target_compile_options(source_disposition_catalog_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(source_disposition_catalog_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(source_disposition_catalog_tests PRIVATE
  trainvm_compatibility_catalog OpenSSL::Crypto
)
add_test(NAME source_disposition_catalog_tests
  COMMAND source_disposition_catalog_tests
)
add_executable(experiment_analysis_tests tests/experiment_analysis_tests.cpp)
target_compile_options(experiment_analysis_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(experiment_analysis_tests PRIVATE
  trainvm_experiment_analysis SQLite3::SQLite3
)
add_test(NAME experiment_analysis_tests COMMAND experiment_analysis_tests)
add_executable(training_schedule_tests tests/training_schedule_tests.cpp)
target_compile_options(training_schedule_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(training_schedule_tests PRIVATE
  trainvm_training_schedules
)
add_test(NAME training_schedule_tests COMMAND training_schedule_tests)
if(TRAINVM_PYTHON_EXECUTABLE)
  add_test(NAME training_schedule_python_parity
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../src
      ${TRAINVM_PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_training_schedule_parity.py
      $<TARGET_FILE:trainvm>
  )
endif()
set(TRAINVM_LEGACY_SOURCE_ROOT "" CACHE PATH
  "Optional original moe-mla root for live source-disposition digest checks"
)
if(TRAINVM_LEGACY_SOURCE_ROOT)
  set_tests_properties(source_disposition_catalog_tests PROPERTIES
    ENVIRONMENT "TRAINVM_LEGACY_SOURCE_ROOT=${TRAINVM_LEGACY_SOURCE_ROOT}"
  )
endif()
add_executable(coverage_fixture_tests tests/coverage_fixture_tests.cpp)
target_compile_options(coverage_fixture_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(coverage_fixture_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(coverage_fixture_tests PRIVATE trainvm_core)
add_test(NAME coverage_fixture_tests COMMAND coverage_fixture_tests)
add_executable(host_resources_tests tests/host_resources_tests.cpp)
target_compile_options(host_resources_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(host_resources_tests PRIVATE trainvm_core)
add_test(NAME host_resources_tests COMMAND host_resources_tests)
add_executable(resource_request_builder_tests
  tests/resource_request_builder_tests.cpp
)
target_compile_options(resource_request_builder_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(resource_request_builder_tests PRIVATE trainvm_core)
add_test(NAME resource_request_builder_tests
  COMMAND resource_request_builder_tests
)
add_executable(training_preflight_tests
  tests/training_preflight_tests.cpp
)
target_compile_options(training_preflight_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(training_preflight_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(training_preflight_tests PRIVATE trainvm_core)
add_test(NAME training_preflight_tests COMMAND training_preflight_tests)
add_executable(run_authoring_tests tests/run_authoring_tests.cpp)
target_compile_options(run_authoring_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(run_authoring_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(run_authoring_tests PRIVATE trainvm_service)
add_test(NAME run_authoring_tests COMMAND run_authoring_tests)
add_executable(author_run_service_tests tests/author_run_service_tests.cpp)
target_compile_options(author_run_service_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(author_run_service_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(author_run_service_tests PRIVATE trainvm_service)
add_test(NAME author_run_service_tests COMMAND author_run_service_tests)
add_executable(lifecycle_equivalence_tests
  tests/lifecycle_equivalence_tests.cpp
)
target_compile_options(lifecycle_equivalence_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(lifecycle_equivalence_tests PRIVATE trainvm_core)
add_test(NAME lifecycle_equivalence_tests
  COMMAND lifecycle_equivalence_tests
)
add_executable(authority_fuzz_tests tests/authority_fuzz_tests.cpp)
target_compile_options(authority_fuzz_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(authority_fuzz_tests PRIVATE trainvm_core)
add_test(NAME authority_fuzz_tests COMMAND authority_fuzz_tests)
add_executable(rwkv_scratch_profiles_tests
  tests/rwkv_scratch_profiles_tests.cpp
)
target_compile_options(rwkv_scratch_profiles_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(rwkv_scratch_profiles_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(rwkv_scratch_profiles_tests PRIVATE trainvm_core)
add_test(NAME rwkv_scratch_profiles_tests
  COMMAND rwkv_scratch_profiles_tests
)
add_executable(profiler_launch_profiles_tests
  tests/profiler_launch_profiles_tests.cpp
)
target_compile_options(profiler_launch_profiles_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(profiler_launch_profiles_tests PRIVATE trainvm_core)
add_test(NAME profiler_launch_profiles_tests
  COMMAND profiler_launch_profiles_tests
)
add_executable(external_profiler_artifact_tests
  tests/external_profiler_artifact_tests.cpp
)
target_compile_options(external_profiler_artifact_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(external_profiler_artifact_tests PRIVATE trainvm_core)
add_test(NAME external_profiler_artifact_tests
  COMMAND external_profiler_artifact_tests
)
add_executable(conversion_publication_tests
  tests/conversion_publication_tests.cpp
)
target_compile_options(conversion_publication_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(conversion_publication_tests PRIVATE trainvm_core)
add_test(NAME conversion_publication_tests
  COMMAND conversion_publication_tests
)
add_executable(post_training_authority_tests
  tests/post_training_authority_tests.cpp
)
target_compile_options(post_training_authority_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(post_training_authority_tests PRIVATE trainvm_core)
add_test(NAME post_training_authority_tests
  COMMAND post_training_authority_tests
)
add_executable(host_ledger_tests tests/host_ledger_tests.cpp)
target_compile_options(host_ledger_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(host_ledger_tests PRIVATE trainvm_core)
add_test(NAME host_ledger_tests COMMAND host_ledger_tests)
add_executable(host_ledger_authority_tests tests/host_ledger_authority_tests.cpp)
target_compile_options(host_ledger_authority_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(host_ledger_authority_tests PRIVATE trainvm_core)
add_test(NAME host_ledger_authority_tests COMMAND host_ledger_authority_tests)
add_executable(sqlite_auxiliary_race_tests
  tests/sqlite_auxiliary_race_tests.cpp
)
target_compile_options(sqlite_auxiliary_race_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(sqlite_auxiliary_race_tests PRIVATE trainvm_core)
add_test(NAME sqlite_auxiliary_race_tests COMMAND sqlite_auxiliary_race_tests)
add_executable(hostd_tests tests/hostd_tests.cpp)
target_compile_options(hostd_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_tests PRIVATE trainvm_core)
add_test(NAME hostd_tests COMMAND hostd_tests)
add_executable(hostd_transport_tests tests/hostd_transport_tests.cpp)
target_compile_options(hostd_transport_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  -fno-access-control
)
target_link_libraries(hostd_transport_tests PRIVATE trainvm_core)
add_test(NAME hostd_transport_tests COMMAND hostd_transport_tests)
add_executable(host_startup_audit_tests tests/host_startup_audit_tests.cpp)
target_compile_options(host_startup_audit_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(host_startup_audit_tests PRIVATE trainvm_core)
add_test(NAME host_startup_audit_tests COMMAND host_startup_audit_tests)
add_executable(hostd_startup_auditor_tests
  tests/hostd_startup_auditor_tests.cpp
)
target_compile_options(hostd_startup_auditor_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_startup_auditor_tests PRIVATE trainvm_core)
add_test(NAME hostd_startup_auditor_tests COMMAND hostd_startup_auditor_tests)
add_executable(hostd_terminal_release_recovery_tests
  tests/hostd_terminal_release_recovery_tests.cpp
)
target_compile_options(hostd_terminal_release_recovery_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_terminal_release_recovery_tests PRIVATE trainvm_core)
add_test(NAME hostd_terminal_release_recovery_tests
  COMMAND hostd_terminal_release_recovery_tests
)
add_executable(hostd_restart_process_recovery_tests
  tests/hostd_restart_process_recovery_tests.cpp
)
target_compile_options(hostd_restart_process_recovery_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_restart_process_recovery_tests PRIVATE trainvm_core)
add_test(NAME hostd_restart_process_recovery_tests
  COMMAND hostd_restart_process_recovery_tests
)
add_executable(hostd_crash_qualification_tests
  tests/hostd_crash_qualification_tests.cpp
)
target_compile_options(hostd_crash_qualification_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_crash_qualification_tests PRIVATE trainvm_core)
add_test(NAME hostd_crash_qualification_tests
  COMMAND hostd_crash_qualification_tests
)
add_executable(hostd_startup_controller_tests
  tests/hostd_startup_controller_tests.cpp
)
target_compile_options(hostd_startup_controller_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_startup_controller_tests PRIVATE trainvm_core)
add_test(NAME hostd_startup_controller_tests
  COMMAND hostd_startup_controller_tests
)
add_executable(hostd_session_challenge_tests
  tests/hostd_session_challenge_tests.cpp
)
target_compile_options(hostd_session_challenge_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_session_challenge_tests PRIVATE trainvm_core)
add_test(NAME hostd_session_challenge_tests COMMAND hostd_session_challenge_tests)
add_executable(hostd_journal_fence_attestor_tests
  tests/hostd_journal_fence_attestor_tests.cpp
)
target_compile_options(hostd_journal_fence_attestor_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_journal_fence_attestor_tests PRIVATE trainvm_core)
add_test(NAME hostd_journal_fence_attestor_tests
  COMMAND hostd_journal_fence_attestor_tests
)
add_executable(hostd_journal_logical_fence_tests
  tests/hostd_journal_logical_fence_tests.cpp
)
target_compile_options(hostd_journal_logical_fence_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_journal_logical_fence_tests PRIVATE trainvm_core)
add_test(NAME hostd_journal_logical_fence_tests
  COMMAND hostd_journal_logical_fence_tests
)
add_executable(hostd_client_configuration_tests
  tests/hostd_client_configuration_tests.cpp
)
target_compile_options(hostd_client_configuration_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_client_configuration_tests PRIVATE trainvm_core)
add_test(NAME hostd_client_configuration_tests
  COMMAND hostd_client_configuration_tests
)
add_executable(hostd_daemon_configuration_tests
  tests/hostd_daemon_configuration_tests.cpp
)
target_compile_options(hostd_daemon_configuration_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_daemon_configuration_tests PRIVATE trainvm_core)
add_test(NAME hostd_daemon_configuration_tests
  COMMAND hostd_daemon_configuration_tests
)
add_executable(hostd_boot_provisioning_tests
  tests/hostd_boot_provisioning_tests.cpp
)
target_compile_options(hostd_boot_provisioning_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_compile_definitions(hostd_boot_provisioning_tests PRIVATE
  TRAINVM_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(hostd_boot_provisioning_tests PRIVATE trainvm_core)
add_test(NAME hostd_boot_provisioning_tests
  COMMAND hostd_boot_provisioning_tests
)
add_executable(hostd_ledger_singleton_token_tests
  tests/hostd_ledger_singleton_token_tests.cpp
)
target_compile_options(hostd_ledger_singleton_token_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_ledger_singleton_token_tests PRIVATE trainvm_core)
add_test(NAME hostd_ledger_singleton_token_tests
  COMMAND hostd_ledger_singleton_token_tests
)
add_executable(hostd_client_bootstrap_tests
  tests/hostd_client_bootstrap_tests.cpp
)
target_compile_options(hostd_client_bootstrap_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_client_bootstrap_tests PRIVATE trainvm_service)
add_test(NAME hostd_client_bootstrap_tests
  COMMAND hostd_client_bootstrap_tests
)
add_executable(hostd_linux_session_authority_tests
  tests/hostd_linux_session_authority_tests.cpp
)
target_compile_options(hostd_linux_session_authority_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_session_authority_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_session_authority_tests
  COMMAND hostd_linux_session_authority_tests
)
add_executable(hostd_linux_service_identity_tests
  tests/hostd_linux_service_identity_tests.cpp
)
target_compile_options(hostd_linux_service_identity_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_service_identity_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_service_identity_tests
  COMMAND hostd_linux_service_identity_tests
)
add_executable(hostd_linux_stopped_launcher_tests
  tests/hostd_linux_stopped_launcher_tests.cpp
)
target_compile_options(hostd_linux_stopped_launcher_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_stopped_launcher_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_stopped_launcher_tests
  COMMAND hostd_linux_stopped_launcher_tests
)
add_executable(hostd_linux_process_recovery_tests
  tests/hostd_linux_process_recovery_tests.cpp
)
target_compile_options(hostd_linux_process_recovery_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_process_recovery_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_process_recovery_tests
  COMMAND hostd_linux_process_recovery_tests
)
add_executable(hostd_linux_inventory_context_auditor_tests
  tests/hostd_linux_inventory_context_auditor_tests.cpp
)
target_compile_options(hostd_linux_inventory_context_auditor_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_inventory_context_auditor_tests PRIVATE
  trainvm_core
)
add_test(NAME hostd_linux_inventory_context_auditor_tests
  COMMAND hostd_linux_inventory_context_auditor_tests
)
add_executable(hostd_linux_cgroup_authority_tests
  tests/hostd_linux_cgroup_authority_tests.cpp
)
target_compile_options(hostd_linux_cgroup_authority_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_cgroup_authority_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_cgroup_authority_tests
  COMMAND hostd_linux_cgroup_authority_tests
)
add_executable(hostd_linux_device_policy_tests
  tests/hostd_linux_device_policy_tests.cpp
)
target_compile_options(hostd_linux_device_policy_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_device_policy_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_device_policy_tests
  COMMAND hostd_linux_device_policy_tests
)
add_executable(hostd_linux_device_kernel_tests
  tests/hostd_linux_device_kernel_tests.cpp
)
target_compile_options(hostd_linux_device_kernel_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  -fno-access-control
)
target_link_libraries(hostd_linux_device_kernel_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_device_kernel_tests
  COMMAND hostd_linux_device_kernel_tests
)
add_executable(hostd_linux_process_policy_tests
  tests/hostd_linux_process_policy_tests.cpp
)
target_compile_options(hostd_linux_process_policy_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_linux_process_policy_tests PRIVATE trainvm_core)
add_test(NAME hostd_linux_process_policy_tests
  COMMAND hostd_linux_process_policy_tests
)
add_executable(hostd_linux_process_policy_kernel_tests
  tests/hostd_linux_process_policy_kernel_tests.cpp
)
target_compile_options(hostd_linux_process_policy_kernel_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  -fno-access-control
)
target_link_libraries(hostd_linux_process_policy_kernel_tests PRIVATE
  trainvm_core
)
add_test(NAME hostd_linux_process_policy_kernel_tests
  COMMAND hostd_linux_process_policy_kernel_tests
)
add_executable(linux_nvidia_inventory_tests
  tests/linux_nvidia_inventory_tests.cpp
)
target_compile_options(linux_nvidia_inventory_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(linux_nvidia_inventory_tests PRIVATE trainvm_core)
add_test(NAME linux_nvidia_inventory_tests COMMAND linux_nvidia_inventory_tests)
add_executable(input_content_authority_tests
  tests/input_content_authority_tests.cpp
)
target_compile_options(input_content_authority_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(input_content_authority_tests PRIVATE trainvm_core)
add_test(NAME input_content_authority_tests COMMAND input_content_authority_tests)
add_executable(input_content_cache_benchmark EXCLUDE_FROM_ALL
  benchmarks/input_content_cache_benchmark.cpp
)
target_compile_options(input_content_cache_benchmark PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(input_content_cache_benchmark PRIVATE trainvm_core)
add_executable(json_optional_tests
  tests/json_optional_tests.cpp
)
target_include_directories(json_optional_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(json_optional_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(json_optional_tests PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME json_optional_tests COMMAND json_optional_tests)
add_test(NAME input_content_authority_python_parity
  COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_input_content_authority_parity.py"
    $<TARGET_FILE:trainvm>
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
add_test(NAME input_content_lock_cli
  COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_input_content_lock.py"
    $<TARGET_FILE:trainvm>
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

get_target_property(TRAINVM_SERVICE_LINKS trainvm_service LINK_LIBRARIES)
if("trainvm_compatibility_catalog" IN_LIST TRAINVM_SERVICE_LINKS)
  message(FATAL_ERROR
    "trainvm_service must not link the compatibility evidence catalog")
endif()
add_test(
  NAME trainvm_cli_validate
  COMMAND trainvm validate
    "${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/mageflow-cache-resume.json"
)
add_test(
  NAME trainvm_cli_simulate
  COMMAND trainvm simulate
    "${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/mageflow-cache-resume.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/mageflow-cache-resume.events.jsonl"
)
add_test(
  NAME trainvm_cli_validate_compatibility_catalog
  COMMAND trainvm validate-catalog
    "${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/compatibility-workflows.v1.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
# The gate must be reachable from a runner and must report a rejection as a
# distinct, non-zero verdict rather than a crash. Both directions are pinned:
# a qualified receipt exits 0, a rejected one exits 3 with attributable reasons.
add_test(
  NAME trainvm_cli_qualify_evidence_qualified
  COMMAND ${CMAKE_COMMAND} -E env
    "TRAINVM_EVIDENCE=${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/qualification-evidence.qualified.v1.json"
    ${CMAKE_COMMAND} -DTRAINVM_BINARY=$<TARGET_FILE:trainvm>
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_qualify_evidence.cmake"
)
set_tests_properties(trainvm_cli_qualify_evidence_qualified PROPERTIES
  PASS_REGULAR_EXPRESSION "\"qualified\": true"
)
add_test(
  NAME trainvm_cli_qualify_evidence_rejected
  COMMAND ${CMAKE_COMMAND} -E env
    "TRAINVM_EVIDENCE=${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/qualification-evidence.rejected.v1.json"
    "TRAINVM_EXPECT_REJECTED=1"
    ${CMAKE_COMMAND} -DTRAINVM_BINARY=$<TARGET_FILE:trainvm>
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_qualify_evidence.cmake"
)
set_tests_properties(trainvm_cli_qualify_evidence_rejected PROPERTIES
  PASS_REGULAR_EXPRESSION "throughput_gate_failed"
)
add_test(
  NAME trainvm_cli_inspect_training_components
  COMMAND trainvm inspect-training-components
    "${CMAKE_CURRENT_SOURCE_DIR}/../docs/experiment-vm/examples/training-components.v1.json"
)
set_tests_properties(trainvm_cli_inspect_training_components PROPERTIES
  PASS_REGULAR_EXPRESSION "rwkv_lab.optimizer.fp32_master_adamw.v1"
)
set_tests_properties(trainvm_cli_validate_compatibility_catalog PROPERTIES
  PASS_REGULAR_EXPRESSION "compatibility_evidence_only"
)
set_tests_properties(trainvm_cli_simulate PROPERTIES
  PASS_REGULAR_EXPRESSION "\"status\": \"completed\""
)
