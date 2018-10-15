cmake_minimum_required(VERSION 2.8)

project(Sort)

############################################################
# Create variables
############################################################

set(INCDIR  ${PROJECT_SOURCE_DIR}/include)
set(LIBDIR  ${PROJECT_SOURCE_DIR}/lib)
set(PACK    sorter util)
set(MAIN    main)

############################################################
# process variables
############################################################

# exe output directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# lib output directory
set(LIBRARY_OUTPUT_PATH ${LIBDIR})

# SRC
foreach(i  ${PACK})
  file(GLOB FILE src/${i}/*.cpp)
  set(SRC ${SRC} ${FILE})
endforeach() 

# MAINDIR
set(MAINDIR src/${MAIN}/)

############################################################
# Create libarary     
############################################################

add_library(LIB STATIC
  ${SRC}
)

target_include_directories(LIB
  PUBLIC 
    INC
)

############################################################
# Create compile flags
############################################################

SET(CMAKE_CXX_FLAGS "-Wall -g")
SET(CMAKE_CXX_FLAGS "-Wall -O3")

############################################################
# Create an executable
############################################################


# Add an executable with the above sources
add_executable(i ${MAINDIR}insertionMain.cpp)
add_executable(m ${MAINDIR}mergeMain.cpp)
add_executable(h ${MAINDIR}heapMain.cpp)
add_executable(q ${MAINDIR}quickMain.cpp)
# link the new hello_library target with the hello_binary target
target_link_libraries(i PRIVATE LIB)
target_link_libraries(m PRIVATE LIB)
target_link_libraries(h PRIVATE LIB)
target_link_libraries(q PRIVATE LIB)


set_target_properties(i PROPERTIES OUTPUT_NAME insertionSort)
set_target_properties(m PROPERTIES OUTPUT_NAME mergeSort)
set_target_properties(h PROPERTIES OUTPUT_NAME heapSort)
set_target_properties(q PROPERTIES OUTPUT_NAME quickSort)
