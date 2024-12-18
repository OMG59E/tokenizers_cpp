cmake_minimum_required(VERSION 3.10)

project(tokenizer_cpp C CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(3rdparty/include)
include_directories(3rdparty/include)
include_directories(${PROJECT_SOURCE_DIR})

link_directories(3rdparty/lib)

list(APPEND LINK_LIBS onnxruntime dl)

add_library(tokenizers SHARED tokenizers.cpp)
target_link_libraries(tokenizers ${LINK_LIBS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} tokenizers)