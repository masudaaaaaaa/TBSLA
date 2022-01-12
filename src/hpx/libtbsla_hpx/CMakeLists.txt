list(APPEND SRC_LIST MatrixCOO.cpp)
list(APPEND SRC_LIST MatrixSCOO.cpp)
list(APPEND SRC_LIST MatrixCSR.cpp)
list(APPEND SRC_LIST MatrixELL.cpp)
list(APPEND SRC_LIST MatrixDENSE.cpp)
list(APPEND SRC_LIST Vector.cpp)

add_library(tbsla_hpx SHARED ${SRC_LIST})
target_include_directories(tbsla_hpx PUBLIC ${PROJECT_SOURCE_DIR}/hpx/include)
target_link_libraries(tbsla_hpx PUBLIC tbsla)
hpx_setup_target(tbsla_hpx COMPONENT_DEPENDENCIES iostreams)
target_include_directories(tbsla_hpx
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/hpx/include>
  PUBLIC $<INSTALL_INTERFACE:include>
)
install(TARGETS tbsla_hpx)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/hpx/include/tbsla DESTINATION "include")
