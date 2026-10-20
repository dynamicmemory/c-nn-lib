# CMake generated Testfile for 
# Source directory: /home/lenny/projects/neuralnet-c
# Build directory: /home/lenny/projects/neuralnet-c/build-asan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(matrix_tests "/home/lenny/projects/neuralnet-c/build-asan/test_matrix")
set_tests_properties(matrix_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/lenny/projects/neuralnet-c/CMakeLists.txt;118;add_test;/home/lenny/projects/neuralnet-c/CMakeLists.txt;0;")
add_test(activation_tests "/home/lenny/projects/neuralnet-c/build-asan/test_activation")
set_tests_properties(activation_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/lenny/projects/neuralnet-c/CMakeLists.txt;119;add_test;/home/lenny/projects/neuralnet-c/CMakeLists.txt;0;")
add_test(loss_tests "/home/lenny/projects/neuralnet-c/build-asan/test_loss")
set_tests_properties(loss_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/lenny/projects/neuralnet-c/CMakeLists.txt;120;add_test;/home/lenny/projects/neuralnet-c/CMakeLists.txt;0;")
add_test(network_tests "/home/lenny/projects/neuralnet-c/build-asan/test_network")
set_tests_properties(network_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/lenny/projects/neuralnet-c/CMakeLists.txt;121;add_test;/home/lenny/projects/neuralnet-c/CMakeLists.txt;0;")
add_test(training_tests "/home/lenny/projects/neuralnet-c/build-asan/test_training")
set_tests_properties(training_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/lenny/projects/neuralnet-c/CMakeLists.txt;122;add_test;/home/lenny/projects/neuralnet-c/CMakeLists.txt;0;")
