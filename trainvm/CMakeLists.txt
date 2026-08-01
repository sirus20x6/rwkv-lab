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
find_package(SQLite3 REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

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
  src/cache_namespace.cpp
  src/control.cpp
  src/controller.cpp
  src/document.cpp
  src/fake_worker.cpp
  src/fsm.cpp
  src/host_ledger.cpp
  src/host_ledger_authority.cpp
  src/host_process_saga.cpp
  src/host_startup_audit.cpp
  src/hostd.cpp
  src/hostd_journal_fence_attestor.cpp
  src/hostd_linux_cgroup_authority.cpp
  src/hostd_linux_process_authority.cpp
  src/hostd_linux_service_identity.cpp
  src/hostd_linux_session_authority.cpp
  src/hostd_linux_stopped_launcher.cpp
  src/hostd_mutation_protocol.cpp
  src/hostd_process_protocol.cpp
  src/hostd_session_challenge.cpp
  src/hostd_transport.cpp
  src/host_launch.cpp
  src/host_launch_registry.cpp
  src/host_resources.cpp
  src/journal.cpp
  src/lease_renewal.cpp
  src/linux_nvidia_inventory.cpp
  src/reconciler.cpp
  src/training_component_registry.cpp
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

add_library(trainvm_service src/service.cpp)
target_include_directories(trainvm_service PUBLIC include)
target_compile_options(trainvm_service PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm_service PUBLIC trainvm_core trainvm_protocol)

add_executable(trainvm src/main.cpp)
target_compile_options(trainvm PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm PRIVATE
  trainvm_service trainvm_compatibility_catalog trainvm_experiment_analysis
)

enable_testing()
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
add_executable(authority_time_tests tests/authority_time_tests.cpp)
target_compile_options(authority_time_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(authority_time_tests PRIVATE trainvm_core)
add_test(NAME authority_time_tests COMMAND authority_time_tests)
add_executable(cache_namespace_tests tests/cache_namespace_tests.cpp)
target_compile_options(cache_namespace_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(cache_namespace_tests PRIVATE trainvm_core)
add_test(NAME cache_namespace_tests COMMAND cache_namespace_tests)
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
add_executable(hostd_tests tests/hostd_tests.cpp)
target_compile_options(hostd_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_tests PRIVATE trainvm_core)
add_test(NAME hostd_tests COMMAND hostd_tests)
add_executable(hostd_transport_tests tests/hostd_transport_tests.cpp)
target_compile_options(hostd_transport_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(hostd_transport_tests PRIVATE trainvm_core)
add_test(NAME hostd_transport_tests COMMAND hostd_transport_tests)
add_executable(host_startup_audit_tests tests/host_startup_audit_tests.cpp)
target_compile_options(host_startup_audit_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(host_startup_audit_tests PRIVATE trainvm_core)
add_test(NAME host_startup_audit_tests COMMAND host_startup_audit_tests)
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
add_executable(linux_nvidia_inventory_tests
  tests/linux_nvidia_inventory_tests.cpp
)
target_compile_options(linux_nvidia_inventory_tests PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(linux_nvidia_inventory_tests PRIVATE trainvm_core)
add_test(NAME linux_nvidia_inventory_tests COMMAND linux_nvidia_inventory_tests)

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
