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
  src/adapter_registry.cpp
  src/authority_time.cpp
  src/cache_namespace.cpp
  src/control.cpp
  src/controller.cpp
  src/document.cpp
  src/fake_worker.cpp
  src/fsm.cpp
  src/host_ledger.cpp
  src/host_ledger_authority.cpp
  src/hostd.cpp
  src/host_launch.cpp
  src/host_launch_registry.cpp
  src/host_resources.cpp
  src/journal.cpp
  src/lease_renewal.cpp
  src/reconciler.cpp
)
target_include_directories(trainvm_core PUBLIC include)
target_compile_options(trainvm_core PUBLIC -freflection)
target_compile_options(trainvm_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm_core
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE OpenSSL::Crypto SQLite3::SQLite3 yaml-cpp::yaml-cpp
)

add_library(trainvm_compatibility_catalog src/compatibility_catalog.cpp)
target_include_directories(trainvm_compatibility_catalog PUBLIC include)
target_compile_options(trainvm_compatibility_catalog PUBLIC -freflection)
target_compile_options(trainvm_compatibility_catalog PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
)
target_link_libraries(trainvm_compatibility_catalog
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE OpenSSL::Crypto
)

add_library(trainvm_service src/service.cpp)
target_include_directories(trainvm_service PUBLIC include)
target_compile_options(trainvm_service PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm_service PUBLIC trainvm_core trainvm_protocol)

add_executable(trainvm src/main.cpp)
target_compile_options(trainvm PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
target_link_libraries(trainvm PRIVATE trainvm_service trainvm_compatibility_catalog)

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
set_tests_properties(trainvm_cli_validate_compatibility_catalog PROPERTIES
  PASS_REGULAR_EXPRESSION "compatibility_evidence_only"
)
set_tests_properties(trainvm_cli_simulate PROPERTIES
  PASS_REGULAR_EXPRESSION "\"status\": \"completed\""
)
