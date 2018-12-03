cmake_minimum_required( VERSION 3.0 )

project( one_pass_morphology )

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

find_package( OpenCV REQUIRED )

include_directories( sources )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( HEADER sources/one_pass_morphology.h )

set( SOURCES sources/one_pass_morphology.cpp )
             
add_library( one_pass_morphology ${HEADER} ${SOURCES} )
target_link_libraries( one_pass_morphology ${OpenCV_LIBS} )

add_subdirectory( test )

set( RELEASE_PROFILING OFF )
# Release with debug info
if( RELEASE_PROFILING )
  message("RELEASE_PROFILING: ON")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
  set_target_properties(one_pass_morphology PROPERTIES LINK_FLAGS_RELEASE "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF")
endif()