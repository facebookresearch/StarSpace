cmake_minimum_required(VERSION 2.8)

project(starspace)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
include_directories(SYSTEM ${CONAN_INCLUDE_DIRS} ../src ../src/utils /usr/local/bin/boost_1_63_0/)
link_directories(${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Add a CMake parameter for choosing a desired Python version
set(EXAMPLE_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the example library")

include(CheckCXXCompilerFlag)

# Set a default build configuration if none is specified. 'MinSizeRel' produces the smallest binaries
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

# Try to autodetect Python (can be overridden manually if needed)
set(Python_ADDITIONAL_VERSIONS 3.4 3.5 3.6 3.7)
if (NOT ${EXAMPLE_PYTHON_VERSION} STREQUAL "")
  find_package(PythonLibs ${EXAMPLE_PYTHON_VERSION} EXACT)
  if (NOT PythonLibs_FOUND)
    find_package(PythonLibs ${EXAMPLE_PYTHON_VERSION} REQUIRED)
  endif()
else()
  find_package(PythonLibs REQUIRED)
endif()

# The above sometimes returns version numbers like "3.4.3+"; the "+" must be removed for the next lines to work
string(REPLACE "+" "" PYTHONLIBS_VERSION_STRING "+${PYTHONLIBS_VERSION_STRING}")

# Uncomment the following line if you will also require a matching Python interpreter
# find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING} EXACT REQUIRED)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

  if (HAS_CPP14_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif (HAS_CPP11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "Unsupported compiler -- at least C++11 support is needed!")
  endif()

  # Enable link time optimization and set the default symbol
  # visibility to hidden (very important to obtain small binaries)
  if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    # Default symbol visibility
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

    # Check for Link Time Optimization support
    CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
    if (HAS_LTO_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    endif()
  endif()
endif()

# Include path for Python header files
include_directories(${PYTHON_INCLUDE_DIR})

# Include path for pybind11 header files -- this may need to be changed depending on your setup
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)

# Create the binding library
add_library(starwrap SHARED
  starspace_pybind.cc
  # ... extra files go here ...
)

# Don't add a 'lib' prefix to the shared library
set_target_properties(starwrap PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(starwrap starspace)

if (WIN32)
  if (MSVC)
    # /bigobj is needed for bigger binding projects due to the limit to 64k
    # addressable sections. /MP enables multithreaded builds (relevant when
    # there are many files).
    set_target_properties(starwrap PROPERTIES COMPILE_FLAGS "/MP /bigobj ")

    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      # Enforce size-based optimization and link time code generation on MSVC
      # (~30% smaller binaries in experiments).
      set_target_properties(starwrap APPEND_STRING PROPERTY COMPILE_FLAGS "/Os /GL ")
      set_target_properties(starwrap APPEND_STRING PROPERTY LINK_FLAGS "/LTCG ")
    endif()
  endif()

  # .PYD file extension on Windows
  set_target_properties(starwrap PROPERTIES SUFFIX ".pyd")

  # Link against the Python shared library
  target_link_libraries(starwrap ${PYTHON_LIBRARY})
elseif (UNIX)
  # It's quite common to have multiple copies of the same Python version
  # installed on one's system. E.g.: one copy from the OS and another copy
  # that's statically linked into an application like Blender or Maya.
  # If we link our plugin library against the OS Python here and import it
  # into Blender or Maya later on, this will cause segfaults when multiple
  # conflicting Python instances are active at the same time (even when they
  # are of the same version).

  # Windows is not affected by this issue since it handles DLL imports
  # differently. The solution for Linux and Mac OS is simple: we just don't
  # link against the Python library. The resulting shared library will have
  # missing symbols, but that's perfectly fine -- they will be resolved at
  # import time.

  # .SO file extension on Linux/Mac OS
  set_target_properties(starwrap PROPERTIES SUFFIX ".so")

  # Strip unnecessary sections of the binary on Linux/Mac OS
  if(APPLE)
    set_target_properties(starwrap PROPERTIES MACOSX_RPATH ".")
    set_target_properties(starwrap PROPERTIES LINK_FLAGS "-undefined dynamic_lookup ")
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_custom_command(TARGET starwrap POST_BUILD COMMAND strip -u -r ${PROJECT_BINARY_DIR}/starwrap.so)
    endif()
  else()
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_custom_command(TARGET starwrap POST_BUILD COMMAND strip ${PROJECT_BINARY_DIR}/starwrap.so)
    endif()
  endif()
endif()
