cmake_minimum_required(VERSION 3.24)
project(multi_GPU CUDA)

set(CMAKE_CUDA_STANDARD 14)
set ( PRJ_SRC_LIST )
set ( PRJ_HEADER_LIST)
set ( PRJ_LIBRARIES )
set ( PRJ_INCLUDE_DIRS )
set ( PRJ_COMPILE_OPTIONS )

set ( CUDA_ENABLE true )
if ( CUDA_ENABLE )
    enable_language( CUDA )
    enable_language( CXX )
    list ( APPEND PRJ_COMPILE_OPTIONS -Xcompiler )
    list ( APPEND PRJ_COMPILE_OPTIONS -fopenmp )
endif()

find_package ( OpenMP )
if ( OpenMP_FOUND )
    list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENMP )
    list ( APPEND PRJ_LIBRARIES ${OpenMP_CXX_LIBRARIES} )
    list ( APPEND PRJ_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS} )
endif ()

file ( GLOB root_header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h" )
file ( GLOB root_header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh" )
file ( GLOB root_header_files "${CMAKE_CURRENT_SOURCE_DIR}/mp2/*.cuh" )
file ( GLOB root_header_files "${CMAKE_CURRENT_SOURCE_DIR}/mp2/*.cu" )
file ( GLOB root_src_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file ( GLOB root_src_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" )

list ( APPEND PRJ_HEADER_LIST ${root_header_files} )
list ( APPEND PRJ_SRC_LIST ${root_src_files} )


add_executable ( ${PROJECT_NAME} ${PRJ_SRC_LIST} ${PRJ_HEADER_LIST} )

target_include_directories ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_INCLUDE_DIRS}
        )



set_target_properties (	${PROJECT_NAME}
        PROPERTIES
        CUDA_ARCHITECTURES "35;50;52;72;75"
        CUDA_SEPARABLE_COMPILATION ON
        )

target_compile_options ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_COMPILE_OPTIONS}
        )

target_link_libraries ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_LIBRARIES}
        )
