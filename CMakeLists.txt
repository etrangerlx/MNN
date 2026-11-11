cmake_minimum_required(VERSION 3.16)
project(infer LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(INFER_BUILD_TESTS "Build unit tests" ON)
option(INFER_BUILD_TOOLS "Build command line tools" ON)

# MNN headers / library used by infer_compare. Auto-detect a local build when
# the caller did not pass them explicitly.
set(MNN_INCLUDE_DIR "/home/leixian/workspaces/projects/AIInfer/MNN/include"
    CACHE PATH "MNN include directory")
if(NOT MNN_BUILD_DIR)
  if(EXISTS "/home/leixian/workspaces/projects/AIInfer/MNN/build/libMNN.so")
    set(MNN_BUILD_DIR "/home/leixian/workspaces/projects/AIInfer/MNN/build")
  else()
    set(MNN_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.toolchain/mnn-build)
  endif()
endif()
set(MNN_BUILD_DIR "${MNN_BUILD_DIR}" CACHE PATH "Directory containing libMNN.so" FORCE)

set(INFER_SRC
  source/core/Tensor.cpp
  source/core/ThreadPool.cpp
  source/core/ModelLoader.cpp
  source/core/Interpreter.cpp
  source/backend/cpu/CPUBackend.cpp
  source/backend/cpu/CPUConvolution.cpp
  source/backend/cpu/CPUPooling.cpp
  source/backend/cpu/CPUUnary.cpp
  source/backend/cpu/CPUShape.cpp
  source/backend/cpu/CPUInnerProduct.cpp
)

add_library(infer STATIC ${INFER_SRC})
target_include_directories(infer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mnn_schema
)
find_package(Threads REQUIRED)
target_link_libraries(infer PUBLIC Threads::Threads)
target_compile_options(infer PRIVATE -O3 -Wall -Wno-unused-function)

if(INFER_BUILD_TOOLS)
  add_executable(infer_bench tools/infer_bench.cpp)
  target_link_libraries(infer_bench PRIVATE infer)

  add_executable(mnn_weighted_model tools/mnn_weighted_model.cpp)
  target_link_libraries(mnn_weighted_model PRIVATE infer)
  target_compile_options(mnn_weighted_model PRIVATE -O2)

  # Optional: compare against MNN's own Interpreter, requires a local libMNN.so.
  if(EXISTS ${MNN_BUILD_DIR}/libMNN.so)
    add_executable(infer_compare tools/infer_compare.cpp)
    target_link_libraries(infer_compare PRIVATE infer)
    target_include_directories(infer_compare PRIVATE ${MNN_INCLUDE_DIR})
    target_link_directories(infer_compare PRIVATE ${MNN_BUILD_DIR})
    target_link_libraries(infer_compare PRIVATE MNN)
    set_target_properties(infer_compare PROPERTIES
      BUILD_RPATH ${MNN_BUILD_DIR})
  endif()
endif()

if(INFER_BUILD_TESTS)
  enable_testing()
  add_executable(infer_test tests/test_ops.cpp)
  target_link_libraries(infer_test PRIVATE infer)
  add_test(NAME infer_test COMMAND infer_test)
endif()
