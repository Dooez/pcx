cmake_minimum_required(VERSION 3.16)
project(pcx VERSION 0.3.0 LANGUAGES CXX)

add_library(pcx STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/par_fft.cpp
)
target_include_directories(pcx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
target_compile_options(pcx PRIVATE -march=native -O3 )
set_target_properties(pcx PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
target_compile_definitions(pcx PUBLIC -DPCX_AVX512)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    add_executable(test_fft)
    set(FULL_FFT_TEST ON)
    set(SPLIT_COMPILATION ON)
    if(${FULL_FFT_TEST})
        if(${SPLIT_COMPILATION})
            target_sources(test_fft PUBLIC
                tests/fft/par_f32_w4.cpp
                tests/fft/par_f32_w8.cpp
                tests/fft/par_f32_w16.cpp
                tests/fft/parc_f32_w4.cpp
                tests/fft/parc_f32_w8.cpp
                tests/fft/parc_f32_w16.cpp
                tests/fft/f32_w4.cpp
                tests/fft/f32_w8.cpp
                tests/fft/f32_w16.cpp
                tests/fft/par_f64_w2.cpp
                tests/fft/par_f64_w4.cpp
                tests/fft/par_f64_w8.cpp
                tests/fft/parc_f64_w2.cpp
                tests/fft/parc_f64_w4.cpp
                tests/fft/parc_f64_w8.cpp
                tests/fft/f64_w2.cpp
                tests/fft/f64_w4.cpp
                tests/fft/f64_w8.cpp
                tests/fft/fft_main.cpp
                tests/fft/common.cpp
            )
        else()
            target_sources(test_fft PUBLIC
                tests/fft/fft_main.cpp
                tests/fft/common.cpp
            )
            target_compile_definitions(test_fft PRIVATE -DNO_SPLIT_COMPILE)
        endif()
        target_compile_definitions(test_fft PUBLIC FULL_FFT_TEST)
        set_property(GLOBAL PROPERTY JOB_POOLS build_jobs=3)
        set_property(TARGET test_fft PROPERTY JOB_POOL_COMPILE build_jobs)
    else()
        set(FFT_NODE_SIZE 8)
        target_sources(test_fft PUBLIC
            tests/fft/parc_f32_w16.cpp
            # tests/fft/par_f32_w16.cpp
            # tests/fft/f32_w16.cpp
            # tests/fft/par_f64_w8.cpp
            # tests/fft/f64_w8.cpp
            # tests/fft/f64_w8.cpp
            tests/fft/common.cpp
            tests/fft/fft_main.cpp
        )
        target_compile_definitions(test_fft PRIVATE -DFFT_NODE_SIZE=${FFT_NODE_SIZE})
    endif()

    add_executable(asm_explorer tests/fft/asm_explorer.cpp)
    add_executable(sort tests/fft/sort.cpp)
    add_executable(test_simd tests/test_simd.cpp)
    add_executable(test_tuples tests/test_tuples.cpp)
    add_executable(test_fft_fin tests/fft/fft_if.cpp tests/fft/common.cpp)
    add_executable(test_w1 tests/fft/fft_w1.cpp tests/fft/common.cpp)

    target_link_libraries(asm_explorer PUBLIC pcx)
    set_target_properties(asm_explorer PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(asm_explorer PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(sort PUBLIC pcx)
    set_target_properties(sort PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(sort PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(test_fft PUBLIC pcx)
    set_target_properties(test_fft PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_fft PRIVATE -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(test_w1 PUBLIC pcx)
    set_target_properties(test_w1 PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_w1 PRIVATE -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(test_fft_fin PUBLIC pcx)
    set_target_properties(test_fft_fin PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_fft_fin PRIVATE -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(test_simd PUBLIC pcx)
    set_target_properties(test_simd PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_simd PRIVATE -save-temps -march=native)

    target_link_libraries(test_tuples PUBLIC pcx)
    set_target_properties(test_tuples PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_tuples PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

    ENABLE_TESTING()
    add_test(NAME test_fft COMMAND test_fft)
endif()

