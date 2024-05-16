#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)
PROJECT(kitti)



add_subdirectory(dlib)

macro(add_program name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib  )
endmacro()


add_program(kitti_train_box_regression)
add_program(kitti_train_detector)
add_program(convert_to_xml)

