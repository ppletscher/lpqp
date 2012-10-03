set(SOURCES_WRAPPER
    mex_lpqp.cpp
    mex_trws.cpp
    mex_smoothdualdecomposition.cpp
    mex_treeinference.cpp
    mex_graphdecomposition.cpp
)

find_program(MEX_CMD mex)
find_program(MEXEXT_CMD mexext)

if(MEX_CMD AND MEXEXT_CMD)

    get_filename_component(MEX_REAL_CMD ${MEX_CMD} ABSOLUTE)
    get_filename_component(MEX_PATH ${MEX_REAL_CMD} PATH)

    get_filename_component(MEXEXT_REAL_CMD ${MEXEXT_CMD} ABSOLUTE)
    get_filename_component(MEXEXT_PATH ${MEXEXT_REAL_CMD} PATH)

    if (MEX_PATH STREQUAL MEXEXT_PATH)
        message("Found MATLAB at: " ${MEX_PATH})

        EXECUTE_PROCESS(COMMAND ${MEXEXT_REAL_CMD} OUTPUT_VARIABLE MEX_EXTENSION OUTPUT_STRIP_TRAILING_WHITESPACE)

        SET(LIB_DEP ${CMAKE_BINARY_DIR}/src/liblpqp.a)

        SET(MEX_FLAGS -largeArrayDims CXX\#${CMAKE_CXX_COMPILER} CC\#${CMAKE_CXX_COMPILER})
        SET(INCLUDE_FLAGS -I${MEX_PATH}/../extern/include
        -I${CMAKE_SOURCE_DIR}/external/liblbfgs/include -I${CMAKE_SOURCE_DIR}/src -I${CMAKE_SOURCE_DIR}/external/trws -I${CMAKE_SOURCE_DIR}/external/eigen -I${CMAKE_SOURCE_DIR}/matlab -I${CMAKE_SOURCE_DIR}/matlab)
        
        # helper object file
        ADD_CUSTOM_COMMAND(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matlab_helpers.o
            COMMAND ${CMAKE_CXX_COMPILER}
            ARGS -fPIC -c ${CMAKE_CURRENT_SOURCE_DIR}/matlab_helpers.cpp ${INCLUDE_FLAGS}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matlab_helpers.cpp ${LIB_DEP}
            COMMENT "Building MEX helpers"
        )

        # build each of the MEX wrappers
        set(wrapperfiles)
        foreach(_file ${SOURCES_WRAPPER})
            get_filename_component(mex_name ${_file} NAME_WE)
            set(mex_file ${CMAKE_CURRENT_BINARY_DIR}/${mex_name}.${MEX_EXTENSION})

            ADD_CUSTOM_COMMAND(
                OUTPUT ${mex_file}
                COMMAND ${MEX_REAL_CMD}
                ARGS ${MEX_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${_file} ${INCLUDE_FLAGS} ${CMAKE_BINARY_DIR}/src/liblpqp.a ${CMAKE_CURRENT_BINARY_DIR}/matlab_helpers.o 
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_file} ${LIB_DEP} ${CMAKE_CURRENT_BINARY_DIR}/matlab_helpers.o
                COMMENT "Building MEX extension ${mex_name}.${MEX_EXTENSION}"
            )
            ADD_CUSTOM_TARGET(${mex_name} ALL DEPENDS ${mex_file})

            list(APPEND wrapperfiles ${mex_file})
        endforeach()

        # install the wrappers and Matlab files
        FILE(GLOB MATLAB_SOURCES *.m)
        INSTALL (
            FILES ${wrapperfiles} ${MATLAB_SOURCES}
            DESTINATION matlab
        )
    else()
        message("The 'mex' and 'mexext' programs have been found in different
        locations. It's likely that one of them is not part of the MATLAB
        instalation. Make sure that the 'bin' directory from the MATLAB
        installation is in PATH")
    endif()
else()
    message("Cannot find MATLAB instalation. Make sure that the 'bin'
    directory from the MATLAB installation is in PATH")
endif()
