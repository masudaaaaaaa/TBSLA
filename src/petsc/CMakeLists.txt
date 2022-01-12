list (APPEND CMAKE_MODULE_PATH "../cmake-modules")
find_package(MPI REQUIRED)
find_package(PETSc COMPONENTS CXX REQUIRED)

add_subdirectory(tests)
add_subdirectory(apps)
add_subdirectory(libtbsla_petsc)
