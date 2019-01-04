cmake_minimum_required(VERSION 3.10)
project(g2o_example)

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
LIST( APPEND CMAKE_MODULE_PATH /home/te6000-16/g2o-20170730_git/cmake_modules)

set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
        /usr/local/opencv-3.4.2/share/OpenCV
        /usr/share/OpenCV)
set(OpenCV_DIR /home/te6000-16/opencv-3.4.2/build)
set( OpenCV_FOUND 1 )

find_package(OpenCV 3.4.2 REQUIRED)

SET( G2O_ROOT /home/te6000-16/g2o-20170730_git)
FIND_PACKAGE(G2O REQUIRED)
find_package( Eigen3 REQUIRED )
find_package( Cholmod )

add_executable(g2o_example main.cpp)
target_link_libraries( g2o_example
        ${OpenCV_LIBS}
        g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
        ${CHOLMOD_LIBRARIES}
        )