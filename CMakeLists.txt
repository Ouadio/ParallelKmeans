# General project settings
cmake_minimum_required(VERSION 3.2)
project(ParallelKmeans VERSION 1.0 LANGUAGES CXX)

# Require C++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Forbidden builds within source directory
# mkdir build && cd build/ && cmake .. (from root source directory)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  string(CONCAT errormsg
    "You cannot build in a source directory (or any directory with "
    "a CMakeLists.txt file). Please make a build subdirectory. Feel "
    "free to remove CMakeCache.txt and CMakeFiles.")
  message(FATAL_ERROR "${errormsg}")
endif()

# Required Packages to be linked
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED) 
find_package(OpenMP REQUIRED)

message(STATUS "OpenCV Include dirs ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV LINK LIBRARIES ${OpenCV_LIBS}")
message(STATUS "Boost Link Lib ${Boost_LIBRARIES}")

# Subdirectories
add_subdirectory(src)
add_subdirectory(apps)
