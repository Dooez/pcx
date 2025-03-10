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

    add_executable(test_fft)
    set(FULL_FFT_TEST OFF)
    if(${FULL_FFT_TEST})
        target_sources(test_fft PUBLIC
            tests/fft/f32_node2.cpp
            tests/fft/f32_node4.cpp
            tests/fft/f32_node8.cpp
            tests/fft/f32_node16.cpp
            tests/fft/f64_node2.cpp
            tests/fft/f64_node4.cpp
            tests/fft/f64_node8.cpp
            tests/fft/f64_node16.cpp
            tests/fft/fft_main.cpp
        )
        target_compile_definitions(test_fft PUBLIC FULL_FFT_TEST)
    else()
        set(FFT_NODE_SIZE 8)
        target_sources(test_fft PUBLIC
            tests/fft/f32_node${FFT_NODE_SIZE}.cpp
            tests/fft/f64_node${FFT_NODE_SIZE}.cpp
            tests/fft/fft_main.cpp
        )
        target_compile_definitions(test_fft PRIVATE -DFFT_NODE_SIZE=${FFT_NODE_SIZE})
    endif()

    add_executable(asm_explorer tests/fft/asm_explorer.cpp)
    add_executable(test_simd tests/test_simd.cpp)
    add_executable(test_tuples tests/test_tuples.cpp)

    target_link_libraries(asm_explorer PUBLIC pcx)
    set_target_properties(asm_explorer PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(asm_explorer PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

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
