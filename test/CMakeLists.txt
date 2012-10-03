include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/external/trws)
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)
include_directories(${CMAKE_SOURCE_DIR}/external/liblbfgs/include)

# gtest
include_directories(${CMAKE_SOURCE_DIR}/external/gtest)
include_directories(${CMAKE_SOURCE_DIR}/external/gtest/include)
add_library(gtest STATIC ${CMAKE_SOURCE_DIR}/external/gtest/src/gtest-all.cc)

# LPQPNPBP
add_executable(LPQPNPBP_test LPQPNPBP_test.cpp)
target_link_libraries(LPQPNPBP_test gtest pthread lpqp)
add_test(LPQPNPBP_test LPQPNPBP_test)

# TreeInference
add_executable(TreeInference_test TreeInference_test.cpp)
target_link_libraries(TreeInference_test gtest pthread lpqp)
add_test(TreeInference_test TreeInference_test)

# SmoothDualDecomposition
add_executable(SmoothDualDecomposition_test SmoothDualDecomposition_test.cpp)
target_link_libraries(SmoothDualDecomposition_test gtest pthread lpqp)
add_test(SmoothDualDecomposition_test SmoothDualDecomposition_test)

# LPQPSDD
add_executable(LPQPSDD_test LPQPSDD_test.cpp)
target_link_libraries(LPQPSDD_test gtest pthread lpqp)
add_test(LPQPSDD_test LPQPSDD_test)

# graph decomposition
add_executable(GraphDecomposition_test GraphDecomposition_test.cpp)
target_link_libraries(GraphDecomposition_test gtest pthread lpqp)
add_test(GraphDecomposition_test GraphDecomposition_test)

# LPQPSDD with different graph decompositions
add_executable(LPQPSDD_decomposition_test LPQPSDD_decomposition_test.cpp)
target_link_libraries(LPQPSDD_decomposition_test gtest pthread lpqp)
add_test(LPQPSDD_decomposition_test LPQPSDD_decomposition_test)

