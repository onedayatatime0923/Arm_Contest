cmake_minimum_required(VERSION 2.8)

project(Test)

############################################################
# Create variables
############################################################
set(BASEPATH    ${PROJECT_SOURCE_DIR})
set(INCPATH     ${BASEPATH}/include)
set(LIBPATH     ${BASEPATH}/lib)
# set(LIBINCPATH )
SET(SRCPATH     ${BASEPATH}/src)
set(PACK        binaryClassifier hmm moveClassifier util)
set(MAIN        main)
############################################################
# Create compile flags
############################################################
SET(CMAKE_CXX_FLAGS "-Wall -g")
SET(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++11")
############################################################
# process variables
############################################################
IF (NOT EXISTS "${INCPATH}")
  FILE(MAKE_DIRECTORY ${INCPATH})
ENDIF()
IF (NOT EXISTS "${LIBPATH}")
  FILE(MAKE_DIRECTORY ${LIBPATH})
ENDIF()
IF (NOT EXISTS "${SRCPATH}")
  FILE(MAKE_DIRECTORY ${SRCPATH})
ENDIF()
# exe output directory
set(EXECUTABLE_OUTPUT_PATH ${BASEPATH})
# include directory
INCLUDE_DIRECTORIES(${SRCPATH} ${INCPATH})
# lib output directory
set(LIBRARY_OUTPUT_PATH ${LIBPATH})
# construct lib
file(GLOB ${LIB} ${LIBPATH}/*.a)
foreach(i  ${LIBINCPATH})
  file(GLOB FILE ${i}/*.a)
  set(LIB ${LIB} ${FILE})
endforeach() 
# SRC
foreach(i  ${PACK})
  file(GLOB FILE ${SRCPATH}/${i}/*.cpp)
  set(SRC ${SRC} ${FILE})
endforeach() 
# MAINPATH
set(MAINPATH ${SRCPATH}/${MAIN}/)
############################################################
# Create libarary     
############################################################
add_library(INC STATIC ${SRC})
############################################################
# Create an executable
############################################################
# Add an executable with the above sources
add_executable(b ${MAINPATH}binaryClassifier.cpp)
add_executable(data ${MAINPATH}data.cpp)
add_executable(dtw ${MAINPATH}dtw.cpp)
add_executable(m ${MAINPATH}moveClassifier.cpp)
add_executable(h ${MAINPATH}hmm.cpp)
# link the new hello_library target with the hello_binary target
target_link_libraries(b PRIVATE INC ${LIB})
target_link_libraries(data PRIVATE INC ${LIB})
target_link_libraries(dtw PRIVATE INC ${LIB})
target_link_libraries(m PRIVATE INC ${LIB})
target_link_libraries(h PRIVATE INC ${LIB})
# link the new hello_library target with the hello_binary target
set_target_properties(b PROPERTIES OUTPUT_NAME Test.binaryClassifier)
set_target_properties(data PROPERTIES OUTPUT_NAME Test.data)
set_target_properties(dtw PROPERTIES OUTPUT_NAME Test.dtw)
set_target_properties(m PROPERTIES OUTPUT_NAME Test.moveClassifier)
set_target_properties(h PROPERTIES OUTPUT_NAME Test.hmm)
