add_executable(tbsla_perf_petsc tbsla_perf_petsc.cpp)
target_link_libraries(tbsla_perf_petsc PRIVATE tbsla_petsc)
install(TARGETS tbsla_perf_petsc DESTINATION bin)
