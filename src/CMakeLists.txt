include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)
include_directories(${CMAKE_SOURCE_DIR}/external/trws)
include_directories(${CMAKE_SOURCE_DIR}/external/liblbfgs/include)

file(GLOB CPP_SOURCES *.cpp)
#file(GLOB CPP_SOURCES_TRWS ${CMAKE_SOURCE_DIR}/external/trws/*.cpp)
file(GLOB CPP_SOURCES_LBFGS ${CMAKE_SOURCE_DIR}/external/liblbfgs/lib/*.c)

add_library(lpqp STATIC ${CPP_SOURCES} ${CPP_SOURCES_TRWS} ${CPP_SOURCES_LBFGS})
set_target_properties(lpqp PROPERTIES COMPILE_FLAGS "-fPIC")

INSTALL(TARGETS lpqp ARCHIVE DESTINATION lib)
