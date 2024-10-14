list(APPEND SRC_LIST MatrixCSR.cpp)
list(APPEND SRC_LIST MatrixCOO.cpp)
list(APPEND SRC_LIST MatrixSCOO.cpp)
list(APPEND SRC_LIST MatrixELL.cpp)
list(APPEND SRC_LIST MatrixDENSE.cpp)
list(APPEND SRC_LIST Matrix.cpp)
list(APPEND SRC_LIST CG.cpp)

message("list SRCLISMPI ${SRC_LIST}")

add_library(tbsla_mpi SHARED ${SRC_LIST})
target_include_directories(tbsla_mpi PUBLIC ${PROJECT_SOURCE_DIR}/mpi/include)
target_link_libraries(tbsla_mpi PUBLIC MPI::MPI_CXX)
target_link_libraries(tbsla_mpi PUBLIC tbsla)
target_include_directories(tbsla_mpi
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mpi/include>
  PUBLIC $<INSTALL_INTERFACE:include>
)
install(TARGETS tbsla_mpi)

if(TBSLA_ENABLE_OMP)
  add_library(tbsla_mpi_omp SHARED ${SRC_LIST})
  target_include_directories(tbsla_mpi_omp PUBLIC ${PROJECT_SOURCE_DIR}/mpi/include)
  target_link_libraries(tbsla_mpi_omp PUBLIC MPI::MPI_CXX)
  target_link_libraries(tbsla_mpi_omp PUBLIC tbsla_omp)
  target_include_directories(tbsla_mpi_omp
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mpi/include>
    PUBLIC $<INSTALL_INTERFACE:include>
  )
  install(TARGETS tbsla_mpi_omp)
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/include/tbsla DESTINATION "include")
