#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.12)
PROJECT(sometest)


include(dlib/cmake)

add_executable(main main.cpp)
target_link_libraries(main dlib::dlib  )

