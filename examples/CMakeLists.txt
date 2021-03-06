cmake_minimum_required(VERSION 2.8.12)
project(examples)

option(CPU_ONLY "If use cpu only" OFF)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)

# set caffe root dir
set(Caffe_DIR /home/liusj/dl/caffe/.build_release)
find_package(Caffe)
include_directories(${Caffe_DIR}/include)

find_package(OpenCV REQUIRED)

include_directories(include
        ${OpenCV_INCLUDE_DIRS}
        )

set(SRC_FILES src/classify.cpp)
add_library(classify SHARED ${SRC_FILES})
target_link_libraries(classify
    ${OpenCV_LIBS}
    ${Caffe_LIBRARIES}
    )

set(TEST_FILES test/main.cpp)
add_executable(classification ${TEST_FILES})
target_link_libraries(classification
        classify)

set(TEST_FILES test/classification_multilabel.cpp)
add_executable(multi_classification ${TEST_FILES})
target_link_libraries(multi_classification
        classify)

set(TOOLS tools/multi_convert_imageset.cpp)
add_executable(multi_convert_imageset ${TOOLS})
target_link_libraries(multi_convert_imageset
        classify)
