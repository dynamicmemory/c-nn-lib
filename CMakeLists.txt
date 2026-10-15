cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(neuronc)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)


add_executable(neuronc 
    src/main.c 
    src/nn.c 
    src/mnist.c 
    src/dummy.c 
)

target_include_directories(neuronc PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(neuronc PRIVATE 
    -Wall 
    -Wextra 
    -Wpedantic 
)

target_link_libraries(neuronc PRIVATE m)
