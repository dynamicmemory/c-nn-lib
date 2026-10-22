cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(nn_lib)
add_compile_definitions(_POSIX_C_SOURCE=199309L)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# Nerual Network library 

add_library(nn_lib STATIC 
    src/network.c 
    src/activation.c 
    src/evaluation.c 
    src/matrix.c 
    src/loss.c 
    src/training.c 
    src/parallel_train.c
)

target_include_directories(nn_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(nn_lib PRIVATE 
    -Wall 
    -Wextra 
    -Wpedantic 
)

target_link_libraries(nn_lib PUBLIC 
    m
    Threads::Threads 
) 

# MNIST Example 

add_executable(mnist
    examples/mnist/main.c 
    examples/mnist/mnist.c 
)

target_compile_options(mnist PRIVATE 
    -Wall 
    -Wextra 
    -Wpedantic 
)

target_link_libraries(mnist PRIVATE 
    nn_lib
)

# Testing

include(CTest)

if(BUILD_TESTING)
    set(TEST_TARGETS
        matrix
        activation
        loss
        network
        training
        evaluation
    )

    add_executable(test_matrix
        test/test_matrix.c
        src/matrix.c
    )

    add_executable(test_activation
        test/test_activation.c
        src/activation.c
    )

    add_executable(test_loss
        test/test_loss.c
        src/loss.c
    )

    add_executable(test_network
        test/test_network.c
        src/network.c
        src/matrix.c
        src/activation.c
        src/loss.c
    )

    add_executable(test_training
        test/test_training.c
        src/training.c
        src/parallel_train.c
        src/network.c
        src/matrix.c
        src/activation.c
        src/loss.c
    )

    add_executable(test_evaluation
        test/test_evaluation.c
        src/evaluation.c
        src/training.c
        src/parallel_train.c
        src/network.c
        src/matrix.c
        src/activation.c
        src/loss.c
    )

    foreach(test_name IN LISTS TEST_TARGETS)
        set(target test_${test_name})

        target_include_directories(${target} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
        )

        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )

        target_link_libraries(${target} PRIVATE m)
        target_link_libraries(${target} PRIVATE pthread)

        add_test(
            NAME ${test_name}_tests
            COMMAND ${target}
        )
    endforeach()
endif()
