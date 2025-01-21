cmake_minimum_required(VERSION 3.16)
project(pcx VERSION 0.3.0 LANGUAGES CXX)
add_library(pcx INTERFACE)
target_include_directories(pcx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
set_target_properties(pcx PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
target_compile_definitions(pcx INTERFACE -DPCX_AVX512)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    add_executable(test_fft tests/test_fft.cpp tests/test_fft_main.cpp)
    add_executable(test_simd tests/test_simd.cpp)
    add_executable(test_tuples tests/test_tuples.cpp)

    target_link_libraries(test_fft PUBLIC pcx)
    set_target_properties(test_fft PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_fft PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(test_simd PUBLIC pcx)
    set_target_properties(test_simd PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_simd PRIVATE -save-temps -march=native)

    target_link_libraries(test_tuples PUBLIC pcx)
    set_target_properties(test_tuples PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_tuples PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

    ENABLE_TESTING()
    add_test(NAME test_fft COMMAND test_fft)
endif()
