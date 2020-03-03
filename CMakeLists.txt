cmake_minimum_required(VERSION 3.3)
project(myFlow)

SET(PYTHON_EXECUTABLE  /home/morzh/anaconda3/bin/python3.7)

find_package(PythonLibs)
find_package(PythonInterp)
find_package(pybind11 REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)


message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

#set(PYBIND11_PYTHON_VERSION 3.7.4)
#set(bindings_python_version 3.7.4)

include_directories( ${pybind11_INCLUDE_DIRS} ${XTL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} )
pybind11_add_module( ${PROJECT_NAME} main.cpp)
target_link_libraries( ${PROJECT_NAME} PRIVATE pybind11::module )
#target_link_libraries( ${PROJECT_NAME} PRIVATE pybind11::module ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES /home/morzh/CLionProjects/pySmoothDeformations/OUTPUT_NAME myFlow)
#target_link_libraries(${targetname} xtl )



#configure_file(InterpolateOpticalFlow.cpython-37m-x86_64-linux-gnu.so /home/morzh/CLionProjects/pySmoothDeformations/InterpolateOpticalFlow.cpython-37m-x86_64-linux-gnu.so COPYONLY)