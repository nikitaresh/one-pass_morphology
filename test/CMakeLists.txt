cmake_minimum_required( VERSION 3.0 )

enable_testing()

find_package( OpenCV REQUIRED )
find_package( GTest REQUIRED )

include_directories( ../sources )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( SOURCES sources/test.cpp )

add_executable( one_pass_morphology_test ${SOURCES} )
target_link_libraries( one_pass_morphology_test one_pass_morphology ${OpenCV_LIBS} GTest::gtest GTest::gtest_main )

add_test( AllTestsInOnePassMorphology one_pass_morphology_test )