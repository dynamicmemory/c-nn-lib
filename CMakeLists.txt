cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(neuronc)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)


add_executable(neuronc 
    src/network.c 
    src/activation.c 
    src/evaluation.c 
    src/matrix.c 
    src/loss.c 
    src/training.c 

    src/main.c 
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
        src/network.c
        src/matrix.c
        src/activation.c
        src/loss.c
    )

    add_executable(test_evaluation
        test/test_evaluation.c
        src/evaluation.c
        src/training.c
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

        add_test(
            NAME ${test_name}_tests
            COMMAND ${target}
        )
    endforeach()
endif()
