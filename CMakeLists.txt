cmake_minimum_required(VERSION 3.16)
project(RFANNS VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

option(RFANNS_NATIVE_ARCH "Enable CPU-specific optimizations for the current machine" ON)

add_library(rfanns STATIC
    src/dataset.cpp
    src/range_tree.cpp
    src/index_core.cpp
    src/index_build.cpp
    src/index_io.cpp
    src/index_search.cpp
    src/index_update.cpp
    src/index_delete.cpp
)

target_compile_features(rfanns PUBLIC cxx_std_17)
target_include_directories(rfanns
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(rfanns SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(rfanns PUBLIC Threads::Threads)
target_compile_options(rfanns PRIVATE -Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND RFANNS_NATIVE_ARCH)
    target_compile_options(rfanns PRIVATE -O3 -march=native -ftree-vectorize)
endif()

add_executable(rfanns_build tools/rfanns_build.cpp)
add_executable(rfanns_search tools/rfanns_search.cpp)
add_executable(rfanns_update tools/rfanns_update.cpp)
add_executable(rfanns_validate tools/rfanns_validate.cpp)
add_executable(generate_filter_groundtruth tools/generate_filter_groundtruth.cpp)

foreach(tool
        rfanns_build
        rfanns_search
        rfanns_update
        rfanns_validate)
    target_link_libraries(${tool} PRIVATE rfanns)
    target_include_directories(${tool} PRIVATE ${PROJECT_SOURCE_DIR}/tools)
    target_compile_options(${tool} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()

target_link_libraries(generate_filter_groundtruth PRIVATE rfanns OpenMP::OpenMP_CXX)
target_include_directories(generate_filter_groundtruth PRIVATE ${PROJECT_SOURCE_DIR}/tools)
target_compile_options(generate_filter_groundtruth PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
    add_executable(rfanns_tests tests/rfanns_tests.cpp)
    target_link_libraries(rfanns_tests PRIVATE rfanns)
    target_include_directories(rfanns_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(rfanns_tests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
    target_compile_options(rfanns_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME rfanns_tests COMMAND rfanns_tests)
endif()
