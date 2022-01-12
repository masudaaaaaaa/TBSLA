list(APPEND SRC_LIST split.cpp InputParser.cpp vector.cpp values_generation.cpp range.cpp csr.cpp cpuset_to_cstr.cpp array.cpp)

add_library(tbsla_utils SHARED ${SRC_LIST})
target_include_directories(tbsla_utils
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
  PUBLIC $<INSTALL_INTERFACE:include>
)
install(TARGETS tbsla_utils)
