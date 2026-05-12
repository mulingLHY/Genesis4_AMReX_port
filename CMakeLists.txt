cmake_minimum_required(VERSION 3.18)
project(Genesis4_AMReX LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ──────────────────────────────────────────────────────
# CUDA architecture selection
# ──────────────────────────────────────────────────────
set(GENESIS_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures: native, 75 for Quadro RTX 6000, 86 for RTX A4000, 89 for RTX 6000 Ada, 90 for H100")

set(CMAKE_CUDA_ARCHITECTURES "${GENESIS_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architectures for Genesis4_AMReX")

# Keep AMReX in sync with this project. Do not FORCE, so users can override it.
set(AMReX_CUDA_ARCH "${GENESIS_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architecture for AMReX")

# Developer options
set(CMAKE_VERBOSE_MAKEFILE ON)
add_compile_options(-g)

# OpenMPI: disable C++ interface
add_compile_definitions(OMPI_SKIP_MPICXX)

# ──────────────────────────────────────────────────────
# External source setup
# ──────────────────────────────────────────────────────
option(G4_AUTO_FETCH_DEPS "Automatically clone AMReX and upstream Genesis4 if missing" ON)
option(G4_RENAME_OVERLAPPED_HEADERS "Rename upstream Genesis4 headers shadowed by this port to *.orig" ON)

set(G4_AMREX_REPOSITORY "https://github.com/AMReX-Codes/amrex.git" CACHE STRING "AMReX git repository")
set(G4_AMREX_GIT_REF "" CACHE STRING "Optional AMReX branch/tag/commit hash. Empty means the repository default branch")
set(G4_AMREX_SOURCE_DIR "${CMAKE_SOURCE_DIR}/AMReX" CACHE PATH "AMReX source directory")

set(G4_UPSTREAM_REPOSITORY "https://github.com/svenreiche/Genesis-1.3-Version4.git" CACHE STRING "Upstream Genesis4 git repository")
set(G4_UPSTREAM_GIT_REF "6d530ac" CACHE STRING "Genesis4 branch/tag/commit hash to clone")
set(G4_UPSTREAM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Genesis-1.3-Version4" CACHE PATH "Upstream Genesis4 source directory")

find_package(Git QUIET)

function(g4_clone_if_missing name repo ref dst required_file)
    if (EXISTS "${dst}/${required_file}")
        message(STATUS "${name}: using existing source at ${dst}")
        return()
    endif()

    if (EXISTS "${dst}")
        message(FATAL_ERROR
            "${name}: ${dst} exists but ${required_file} was not found. "
            "Remove the incomplete directory or set the corresponding *_SOURCE_DIR cache variable.")
    endif()

    if (NOT G4_AUTO_FETCH_DEPS)
        message(FATAL_ERROR
            "${name}: source not found at ${dst}. "
            "Either enable G4_AUTO_FETCH_DEPS or provide the source directory manually.")
    endif()

    if (NOT GIT_FOUND)
        message(FATAL_ERROR "${name}: git executable not found; cannot auto-fetch ${repo}")
    endif()

    get_filename_component(_dst_parent "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_parent}")

    # set(_clone_cmd "${GIT_EXECUTABLE}" clone --depth 1)
    # if (NOT "${ref}" STREQUAL "")
    #     list(APPEND _clone_cmd --branch "${ref}")
    # endif()
    # list(APPEND _clone_cmd "${repo}" "${dst}")

    # message(STATUS "${name}: cloning ${repo} into ${dst}")
    # execute_process(
    #     COMMAND ${_clone_cmd}
    #     RESULT_VARIABLE _clone_result
    #     OUTPUT_VARIABLE _clone_output
    #     ERROR_VARIABLE _clone_error
    # )

    # if (NOT _clone_result EQUAL 0)
    #     message(FATAL_ERROR
    #         "${name}: git clone failed with exit code ${_clone_result}\n"
    #         "Command: ${_clone_cmd}\n"
    #         "Output: ${_clone_output}\n"
    #         "Error: ${_clone_error}")
    # endif()
    set(_clone_cmd "${GIT_EXECUTABLE}" clone)

    if ("${ref}" STREQUAL "")
        list(APPEND _clone_cmd --depth 1 "${repo}" "${dst}")
    else()
        # Do not use --branch here, because ref may be a commit SHA.
        list(APPEND _clone_cmd "${repo}" "${dst}")
    endif()

    message(STATUS "${name}: cloning ${repo} into ${dst}")
    execute_process(
        COMMAND ${_clone_cmd}
        RESULT_VARIABLE _clone_result
        OUTPUT_VARIABLE _clone_output
        ERROR_VARIABLE _clone_error
    )

    if (NOT _clone_result EQUAL 0)
        message(FATAL_ERROR
            "${name}: git clone failed with exit code ${_clone_result}\n"
            "Command: ${_clone_cmd}\n"
            "Output: ${_clone_output}\n"
            "Error: ${_clone_error}")
    endif()

    if (NOT "${ref}" STREQUAL "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${dst}" checkout "${ref}"
            RESULT_VARIABLE _checkout_result
            OUTPUT_VARIABLE _checkout_output
            ERROR_VARIABLE _checkout_error
        )

        if (NOT _checkout_result EQUAL 0)
            message(FATAL_ERROR
                "${name}: git checkout ${ref} failed with exit code ${_checkout_result}\n"
                "Output: ${_checkout_output}\n"
                "Error: ${_checkout_error}")
        endif()
    endif()
    ###### replace after Genesis tag v4.6.13

    if (NOT EXISTS "${dst}/${required_file}")
        message(FATAL_ERROR
            "${name}: clone completed, but ${required_file} was not found in ${dst}")
    endif()
endfunction()

function(g4_preserve_shadowed_genesis_headers overlay_include_dir upstream_include_dir)
    if (NOT G4_RENAME_OVERLAPPED_HEADERS)
        return()
    endif()

    if (NOT IS_DIRECTORY "${overlay_include_dir}")
        return()
    endif()

    if (NOT IS_DIRECTORY "${upstream_include_dir}")
        message(FATAL_ERROR "Genesis4 include directory not found: ${upstream_include_dir}")
    endif()

    file(GLOB _overlay_headers CONFIGURE_DEPENDS
        "${overlay_include_dir}/*.h"
        "${overlay_include_dir}/*.hh"
        "${overlay_include_dir}/*.hpp"
        "${overlay_include_dir}/*.hxx"
    )

    foreach(_overlay_header IN LISTS _overlay_headers)
        get_filename_component(_header_name "${_overlay_header}" NAME)
        set(_upstream_header "${upstream_include_dir}/${_header_name}")
        set(_orig_header "${_upstream_header}.orig")

        if (EXISTS "${_upstream_header}")
            if (EXISTS "${_orig_header}")
                message(WARNING
                    "Genesis4 header ${_upstream_header} is shadowed, but ${_orig_header} already exists. "
                    "Leaving both files untouched to avoid overwriting the preserved original.")
            else()
                file(RENAME "${_upstream_header}" "${_orig_header}")
                message(STATUS "Genesis4 header preserved: include/${_header_name} -> include/${_header_name}.orig")
            endif()
        elseif (EXISTS "${_orig_header}")
            message(STATUS "Genesis4 header already preserved: include/${_header_name}.orig")
        endif()
    endforeach()
endfunction()

g4_clone_if_missing("AMReX" "${G4_AMREX_REPOSITORY}" "${G4_AMREX_GIT_REF}" "${G4_AMREX_SOURCE_DIR}" "CMakeLists.txt")
g4_clone_if_missing("Genesis4" "${G4_UPSTREAM_REPOSITORY}" "${G4_UPSTREAM_GIT_REF}" "${G4_UPSTREAM_SOURCE_DIR}" "include/version.h.in")

# ──────────────────────────────────────────────────────
# Genesis4 source paths
# ──────────────────────────────────────────────────────
set(G4 ${G4_UPSTREAM_SOURCE_DIR})
g4_preserve_shadowed_genesis_headers("${CMAKE_SOURCE_DIR}/include" "${G4}/include")

# ──────────────────────────────────────────────────────
# AMReX configuration (must be set BEFORE add_subdirectory)
# ──────────────────────────────────────────────────────
set(AMReX_GPU_BACKEND "CUDA" CACHE STRING "GPU backend" FORCE)
# set(AMReX_CUDA_ARCH "86" CACHE STRING "CUDA architecture (sm_86 = RTX A4000)" FORCE)
set(AMReX_MPI ON CACHE BOOL "Enable MPI" FORCE)
set(AMReX_OMP OFF CACHE BOOL "Disable OpenMP" FORCE)
set(AMReX_PARTICLES ON CACHE BOOL "Enable particles" FORCE)
set(AMReX_FORTRAN OFF CACHE BOOL "Disable Fortran" FORCE)
set(AMReX_LINEAR_SOLVERS ON CACHE BOOL "Enable linear solvers" FORCE)
set(AMReX_PRECISION "DOUBLE" CACHE STRING "Double precision" FORCE)
add_subdirectory(${G4_AMREX_SOURCE_DIR} ${CMAKE_BINARY_DIR}/AMReX-build)

# ──────────────────────────────────────────────────────
# Find dependencies
# ──────────────────────────────────────────────────────
find_package(MPI REQUIRED)

SET(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)
if (HDF5_IS_PARALLEL)
    message(STATUS "Parallel HDF5 supported")
else()
    message(STATUS "WARNING: No parallel HDF5 support")
endif()

# ──────────────────────────────────────────────────────
# Generate version.h
# ──────────────────────────────────────────────────────
set(_GENESIS_VERSION_MAJOR 4)
set(_GENESIS_VERSION_MINOR 6)
set(_GENESIS_VERSION_REV 2)
set(_GENESIS_VERSION_BETA false)

execute_process(
    COMMAND git -C ${G4} describe --tags --always
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_HASH)
    set(GIT_HASH "unknown")
endif()

execute_process(
    COMMAND whoami
    OUTPUT_VARIABLE _user_name
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(TIMESTAMP _config_time "%Y-%m-%d %H:%M:%S")

configure_file(
    ${G4}/include/version.h.in
    ${CMAKE_BINARY_DIR}/include/version.h
    @ONLY
)

# Files reused directly from Genesis4 (no modification)
set(G4_UNCHANGED_SRC
    # Util
    ${G4}/src/Util/BesselJ.cpp
    ${G4}/src/Util/GaussHermite.cpp
    ${G4}/src/Util/Hammerslay.cpp
    ${G4}/src/Util/Inverfc.cpp
    ${G4}/src/Util/RandomU.cpp
    ${G4}/src/Util/SimpleHandshake.cpp
    ${G4}/src/Util/Sorting.cpp
    ${G4}/src/Util/StringProcessing.cpp
    ${G4}/src/Util/SemaFile.cpp
    ${G4}/src/Util/FFTObj.cpp
    # Lattice
    ${G4}/src/Lattice/AlterLattice.cpp
    ${G4}/src/Lattice/Lattice.cpp
    ${G4}/src/Lattice/LatticeElements.cpp
    ${G4}/src/Lattice/LatticeParser.cpp
    ${G4}/src/Lattice/Optics.cpp
    # Core (unchanged)
    ${G4}/src/Core/Undulator.cpp
    ${G4}/src/Core/Incoherent.cpp
    ${G4}/src/Core/Collective.cpp
    ${G4}/src/Core/DiagnosticUser.cpp
    ${G4}/src/Core/FieldSolverFFT.cpp
    ${G4}/src/Core/EFieldSolver.cpp
    # ${G4}/src/Core/Diagnostic.cpp
    # IO
    ${G4}/src/IO/HDF5base.cpp
    ${G4}/src/IO/Output.cpp
    ${G4}/src/IO/readBeamHDF5.cpp
    ${G4}/src/IO/readFieldHDF5.cpp
    ${G4}/src/IO/SDDSBeam.cpp
    ${G4}/src/IO/writeBeamHDF5.cpp
    ${G4}/src/IO/writeFieldHDF5.cpp
    ${G4}/src/IO/writeWakeHDF5.cpp
    ${G4}/src/IO/readMapHDF5.cpp
    # Loading
    ${G4}/src/Loading/ImportBeam.cpp
    ${G4}/src/Loading/ImportField.cpp
    ${G4}/src/Loading/LoadBeam.cpp
    ${G4}/src/Loading/LoadField.cpp
    ${G4}/src/Loading/GenProfile.cpp
    ${G4}/src/Loading/QuietLoading.cpp
    ${G4}/src/Loading/Series.cpp
    ${G4}/src/Loading/ShotNoise.cpp
    ${G4}/src/Loading/SeriesManager.cpp
    ${G4}/src/Loading/SeriesParser.cpp
    ${G4}/src/Loading/Sequence.cpp
    ${G4}/src/Loading/ImportTransformation.cpp
    # Main
    ${G4}/src/Main/AlterSetup.cpp
    ${G4}/src/Main/Dump.cpp
    ${G4}/src/Main/AlterBeam.cpp
    ${G4}/src/Main/EField.cpp
    ${G4}/src/Main/FieldManipulator.cpp
    ${G4}/src/Main/GenMain.cpp
    ${G4}/src/Main/Parser.cpp
    ${G4}/src/Main/Setup.cpp
    ${G4}/src/Main/SponRad.cpp
    ${G4}/src/Main/GenTime.cpp
    ${G4}/src/Main/Track.cpp
    ${G4}/src/Main/Wake.cpp
)

# AMReX-ported source files (new implementations)
set(AMREX_PORTED_SRC
    src/Core/Gencore.cpp
    src/Core/Beam.cpp
    src/Core/Field.cpp
    src/Core/TrackBeam.cpp
    src/Core/BeamSolver.cpp
    src/Core/FieldSolverADI.cpp
    src/Core/Control.cpp
    src/Core/Diagnostic.cpp
)

# ──────────────────────────────────────────────────────
# Build library
# ──────────────────────────────────────────────────────
add_library(genesis4_amrex STATIC
    ${G4_UNCHANGED_SRC}
    ${AMREX_PORTED_SRC}
)
set_target_properties(genesis4_amrex PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compile AMReX-ported sources as CUDA (they include AMReX headers)
set_source_files_properties(${AMREX_PORTED_SRC} PROPERTIES LANGUAGE CUDA)
# Unchanged Genesis4 sources compile as regular C++ (PIMPL hides AMReX types from headers)

# Include path order: our include/ first (overrides), then Genesis4's, then generated
target_include_directories(genesis4_amrex
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${G4}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(genesis4_amrex
    PUBLIC
    amrex
    MPI::MPI_C
)

target_compile_definitions(genesis4_amrex PUBLIC GENESIS_USE_AMREX=1)

if (TARGET HDF5::HDF5)
    target_link_libraries(genesis4_amrex PUBLIC HDF5::HDF5)
else()
    target_include_directories(genesis4_amrex PUBLIC ${HDF5_C_INCLUDE_DIRS})
    target_link_libraries(genesis4_amrex PUBLIC ${HDF5_C_LIBRARIES})
endif()

# ──────────────────────────────────────────────────────
# Build executable
# ──────────────────────────────────────────────────────
add_executable(genesis4 ${G4}/src/Main/mainwrap.cpp)
target_link_libraries(genesis4 genesis4_amrex)


