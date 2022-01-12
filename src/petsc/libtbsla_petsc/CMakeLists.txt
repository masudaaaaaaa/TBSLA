list(APPEND SRC_LIST Matrix.cpp)

add_library(tbsla_petsc OBJECT ${SRC_LIST})
target_include_directories(tbsla_petsc PUBLIC ${PROJECT_SOURCE_DIR}/petsc/include)
target_include_directories(tbsla_petsc PUBLIC ${PETSC_INCLUDES})
target_link_libraries(tbsla_petsc PUBLIC MPI::MPI_CXX)
target_link_libraries(tbsla_petsc PUBLIC tbsla)
target_link_libraries(tbsla_petsc PUBLIC ${PETSC_LIBRARIES})
