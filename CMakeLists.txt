cmake_minimum_required(VERSION 3.28)
project(parallel_DCE_qsl)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_LIB "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
add_executable(run_evolution run_evolution.cpp evolution/evolution.cpp evolution/evolution.hpp)
add_executable(example fftw_example.cpp)
target_compile_options(run_evolution PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
)
target_compile_options(example PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
)
set_source_files_properties(
        fftw_example.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)

set_source_files_properties(
        evolution/evolution.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)

set_source_files_properties(
        run_evolution.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)

message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")

set(Python3_INCLUDE_DIRS "/home/adada/anaconda3/include/python3.9")
set(Python3_LIBRARIES "/home/adada/anaconda3/lib/libpython3.9.so")
include_directories(${Python3_INCLUDE_DIRS})

set(Boost_NO_SYSTEM_PATHS YES)
if (Boost_NO_SYSTEM_PATHS)
    set(Boost_ROOT "/home/adada/cppLocal/boostBuild")
    set(Boost_INCLUDE_DIR "/home/adada/cppLocal/boostBuild/include")
    set(Boost_LIBRARY_DIR "/home/adada/cppLocal/boostBuild/lib")
endif (Boost_NO_SYSTEM_PATHS)

find_package(Boost REQUIRED COMPONENTS python numpy filesystem)

if(Boost_FOUND)
    message(STATUS "boost root is: ${Boost_ROOT}")
    message(STATUS "boost include path is : ${Boost_INCLUDE_DIR}")
    message(STATUS "boost library path is : ${Boost_LIBRARY_DIR}")
    message(STATUS "boost libraries is : ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
else()
    message(WARNING "boost not found.")
endif()


target_link_libraries(run_evolution Boost::filesystem)

target_link_libraries(run_evolution Boost::python)
target_link_libraries(run_evolution Boost::numpy)
#python
target_link_libraries(run_evolution ${Python3_LIBRARIES})

include_directories("/home/adada/cppLocal/fftw3/include")
include_directories("/home/adada/cppLocal/armadillo/include")
#armadillo
file(GLOB LIBS_SO "/home/adada/cppLocal/armadillo/lib/libarmadillo.so")
target_link_libraries(run_evolution ${LIBS_SO})


#fftw3
file(GLOB LIBS_A "/home/adada/cppLocal/fftw3/lib/libfftw3.a")
target_link_libraries(run_evolution ${LIBS_A})



##### example
target_link_libraries(example Boost::filesystem)

target_link_libraries(example Boost::python)
target_link_libraries(example Boost::numpy)
#python
target_link_libraries(example ${Python3_LIBRARIES})


#armadillo
file(GLOB LIBS_SO "/home/adada/cppLocal/armadillo/lib/libarmadillo.so")
target_link_libraries(example ${LIBS_SO})


#fftw3
file(GLOB LIBS_A "/home/adada/cppLocal/fftw3/lib/*.a")

target_link_libraries(example ${LIBS_A})