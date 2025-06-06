cmake_minimum_required(VERSION 3.25)
project(pcx
    VERSION 0.5.0
    LANGUAGES CXX
)
add_library(pcx)
add_library(pcx::pcx ALIAS pcx)
# add_library(pcx_header_only)
# add_library(pcx::pcx_header_only ALIAS pcx_header_only)

string(COMPARE EQUAL
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    is_top_level)

option(PCX_AVX512 "Compile with avx512 support" ON)

if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
endif()
if(NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN )
    set(CMAKE_VISIBILITY_INLINES_HIDDEN  ON)
endif()


# ---
# Sources
# ---
set(public_headers
    include/pcx/fft.hpp
)
set(sources
    ${public_headers}
    src/fft.cpp
    src/par_fft.cpp
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

# ---
# target config
# ---
include(CMakePackageConfigHelpers)

target_sources(pcx PRIVATE ${sources})
target_include_directories(pcx
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    # PRIVATE
    #     "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set_target_properties(pcx PROPERTIES
    PUBLIC_HEADER "${public_headers}"
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED True
)
target_compile_options(pcx PRIVATE -march=native #[[ -O3 ]])

# ---
# target config
# ---




# add_library(pcx STATIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/fft.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/par_fft.cpp
# )
# target_include_directories(pcx PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     )
target_compile_options(pcx PRIVATE -march=native #[[ -O3 ]])
target_compile_definitions(pcx PUBLIC -DPCX_AVX512)

add_library(pcx_ho INTERFACE)
target_include_directories(pcx_ho INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
target_compile_definitions(pcx_ho INTERFACE -DPCX_AVX512)

set_target_properties(pcx_ho PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    add_executable(test_fft)
    set(F32_WIDTHS 1 2 4 8 16)
    set(F64_WIDTHS 1 2 4 8)
    set(NODE_SIZES 16 8 4 2)
    # set(F32_WIDTHS 16)
    # set(F64_WIDTHS 8)
    # set(F32_WIDTHS 16)
    # set(F64_WIDTHS )
    # set(NODE_SIZES 16)
    set(TEST_PARC ON)
    set(TEST_PAR ON)
    set(TEST_SEQ ON)

    set(TEST_FFT_SOURCES "")
    function(generate_tests PROTOTYPE TYPE WIDTHS NODE_SIZES)
        foreach(NS ${NODE_SIZES})
            set(NODE_SIZE ${NS})
            foreach(WIDTH ${WIDTHS})
                set(L_FILE  "test_sources/${TYPE}_${PROTOTYPE}_ns${NS}_w${WIDTH}.cpp")
                set(FLOAT_TYPE ${TYPE})
                set(VECTOR_WIDTH ${WIDTH})
                configure_file("tests/fft/prototype_${PROTOTYPE}.cpp" "${L_FILE}")
                set(TEST_FFT_SOURCES "${TEST_FFT_SOURCES}" "${L_FILE}")
            endforeach()
        endforeach()
        return(PROPAGATE TEST_FFT_SOURCES)
    endfunction()
    if (${TEST_SEQ})
        generate_tests(seq f32 "${F32_WIDTHS}" "${NODE_SIZES}")
        generate_tests(seq f64 "${F64_WIDTHS}" "${NODE_SIZES}")
    endif()
    if (${TEST_PAR})
        generate_tests(par f32 "${F32_WIDTHS}" "${NODE_SIZES}")
        generate_tests(par f64 "${F64_WIDTHS}" "${NODE_SIZES}")
    endif()
    if (${TEST_PARC})
        generate_tests(parc f32 "${F32_WIDTHS}" "${NODE_SIZES}")
        generate_tests(parc f64 "${F64_WIDTHS}" "${NODE_SIZES}")
    endif()
    set(TEST_FFT_SOURCES "${TEST_FFT_SOURCES}"
        tests/fft/common.cpp
        tests/fft/fft_main.cpp
    )
    target_sources(test_fft PUBLIC "${TEST_FFT_SOURCES}")
    string(REPLACE ";" "," NODE_SIZES "${NODE_SIZES}")
    string(REPLACE ";" "," F32_WIDTHS "${F32_WIDTHS}")
    string(REPLACE ";" "," F64_WIDTHS "${F64_WIDTHS}")
    target_compile_definitions(test_fft PUBLIC
        "-DPCX_TESTING_NODE_SIZES=${NODE_SIZES}"
        "-DPCX_TESTING_F32_WIDTHS=${F32_WIDTHS}"
        "-DPCX_TESTING_F64_WIDTHS=${F64_WIDTHS}"
        "-DPCX_TESTING_SEQ=$<BOOL:${TEST_SEQ}>"
        "-DPCX_TESTING_PAR=$<BOOL:${TEST_PAR}>"
        "-DPCX_TESTING_PARC=$<BOOL:${TEST_PARC}>"
    )
    set_property(GLOBAL PROPERTY JOB_POOLS build_jobs=8)
    set_property(TARGET test_fft PROPERTY JOB_POOL_COMPILE build_jobs)

    add_executable(asm_explorer tests/fft/asm_explorer.cpp)
    add_executable(test_simd tests/test_simd.cpp)
    add_executable(test_tuples tests/test_tuples.cpp)
    add_executable(test_fft_fin tests/fft/fft_if.cpp tests/fft/common.cpp)
    target_compile_definitions(test_fft_fin PUBLIC
        "-DPCX_TESTING_NODE_SIZES=${NODE_SIZES}"
    )

    target_link_libraries(asm_explorer PUBLIC pcx)
    set_target_properties(asm_explorer PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(asm_explorer PRIVATE -save-temps -march=native -ftemplate-backtrace-limit=0)

    target_link_libraries(test_fft PUBLIC pcx_ho)
    set_target_properties(test_fft PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)
    target_compile_options(test_fft PRIVATE -march=native -ftemplate-backtrace-limit=0)

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

