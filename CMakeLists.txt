project(lpqp)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS " -O3 -Wall")
#set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -Wall")
#set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -Wall -g")
#set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -lprofiler")

enable_testing()

# tell cmake to process the sub-directories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(matlab)
