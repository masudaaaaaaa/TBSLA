cmake_minimum_required(VERSION 3.9...3.14)

project(TBSLA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file("${PROJECT_SOURCE_DIR}/includes/tbsla/Configs.h.in" "${PROJECT_SOURCE_DIR}/cpp/include/tbsla/Configs.h")

include(CTest)
enable_testing()

if(TBSLA_ENABLE_OMP)
  find_package(OpenMP REQUIRED)
endif()

add_subdirectory(cpp)

if(TBSLA_ENABLE_MPI)
    add_subdirectory(mpi)
endif()

if(TBSLA_ENABLE_HPX)
    add_subdirectory(hpx)
endif()

if(TBSLA_ENABLE_PETSC)
    add_subdirectory(petsc)
endif()
