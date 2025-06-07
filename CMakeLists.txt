cmake_minimum_required(VERSION 3.25)
project(pcx
    VERSION 0.5.0
    LANGUAGES CXX
)

# general settings

string(COMPARE EQUAL
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    is_top_level)

option(PCX_AVX512 "Compile with avx512 support" ON)
option(PCX_INSTALL "Generate target for installing mylib" ${is_top_level})
option(PCX_BUILD_TESTS OFF)

if(NOT DEFINED PCX_SIMD_BACKEND)
    set(PCX_SIMD_BACKEND "x86_instrinsics"
        CACHE
        STRING "Backend for pcx SIMD instructions")
endif()
if(NOT DEFINED PCX_INSTALL_CMAKEDIR )
    set(PCX_INSTALL_CMAKEDIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/pcx"
        CACHE
        STRING "Install path for pcx package-related CMake files"
    )
endif()
if(DEFINED PCX_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${PCX_SHARED_LIBS})
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
endif()
if(NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN  ON)
endif()

add_library(pcx)
add_library(pcx::pcx ALIAS pcx)
# add_library(pcx_header_only)
# add_library(pcx::pcx_header_only ALIAS pcx_header_only)

# ---
# Sources
# ---
set(public_headers
    include/pcx/fft.hpp
    include/pcx/include/fft/fft_impl.hpp
    include/pcx/include/fft/permuters.hpp
    include/pcx/include/fft/util.hpp
    include/pcx/include/simd/common.hpp
    include/pcx/include/simd/math.hpp
    include/pcx/include/simd/traits.hpp
    include/pcx/include/meta.hpp
    include/pcx/include/tupi.hpp
    include/pcx/include/tuple.hpp
    include/pcx/include/types.hpp
)
if(PCX_BACKEND EQUAL "x86_intrinsics")
    set(public_headers
        ${public_headers}
        include/pcx/include/simd/x86/f32-traits.hpp
        include/pcx/include/simd/x86/f64-traits.hpp
    )
endif()

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
        # "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
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
target_compile_options(pcx PRIVATE -march=native)
if(PCX_AVX512)
    target_compile_definitions(pcx PUBLIC -DPCX_AVX512)
endif()

# ---
# target config
# ---

include(CMakePackageConfigHelpers)
if(PCX_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    configure_package_config_file(cmake/pcx-config.cmake.in
        pcx-config.cmake
        INSTALL_DESTINATION "${PCX_INSTALL_CMAKEDIR}"
    )
    write_basic_package_version_file(pcx-config-version.cmake
        COMPATIBILITY SameMajorVersion
    )
    install(TARGETS pcx
        EXPORT pcx_export
        RUNTIME COMPONENT pcx
        LIBRARY COMPONENT pcx
        NAMELINK_COMPONENT pcx-dev
        ARCHIVE COMPONENT pcx-dev
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    set(targets_file "pcx-shared-targets.cmake")
    if(NOT BUILD_SHARED_LIBS)
        set(targets_file "pcx-static-targets.cmake")
    endif()

    install(EXPORT pcx_export
        COMPONENT pcx-dev
        FILE "${targets_file}"
        DESTINATION "${PCX_INSTALL_CMAKEDIR}"
        NAMESPACE pcx::
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/pcx-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/pcx-config-version.cmake"
        COMPONENT pcx-dev
        DESTINATION "${PCX_INSTALL_CMAKEDIR}"
    )
endif()

# add_library(pcx STATIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/fft.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/par_fft.cpp
# )
# target_include_directories(pcx PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     )

add_library(pcx_ho INTERFACE)
target_include_directories(pcx_ho INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
target_compile_definitions(pcx_ho INTERFACE -DPCX_AVX512)

set_target_properties(pcx_ho PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED True)

if(${is_top_level} AND ${PCX_BUILD_TESTS})
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

