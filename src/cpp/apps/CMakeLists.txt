add_executable(tbsla_gen_matrix tbsla_gen_matrix.cpp)
target_link_libraries(tbsla_gen_matrix PRIVATE tbsla)
install(TARGETS tbsla_gen_matrix DESTINATION bin)

add_executable(tbsla_mm_to_binCOO tbsla_mm_to_binCOO.cpp)
target_link_libraries(tbsla_mm_to_binCOO PRIVATE tbsla)
install(TARGETS tbsla_mm_to_binCOO DESTINATION bin)

add_executable(tbsla_storage_conversion tbsla_storage_conversion.cpp)
target_link_libraries(tbsla_storage_conversion PRIVATE tbsla)
install(TARGETS tbsla_storage_conversion DESTINATION bin)

add_executable(tbsla_print_binCOO_as_Dense tbsla_print_binCOO_as_Dense.cpp)
target_link_libraries(tbsla_print_binCOO_as_Dense PRIVATE tbsla)
install(TARGETS tbsla_print_binCOO_as_Dense DESTINATION bin)

add_executable(tbsla_check_yml_results tbsla_check_yml_results.cpp)
target_link_libraries(tbsla_check_yml_results PRIVATE tbsla)
install(TARGETS tbsla_check_yml_results DESTINATION bin)

add_executable(tbsla_perf tbsla_perf.cpp)
target_link_libraries(tbsla_perf PRIVATE tbsla)
install(TARGETS tbsla_perf DESTINATION bin)

add_executable(test_impl_csr test_impl_csr.cpp)
install(TARGETS test_impl_csr DESTINATION bin)

if(TBSLA_ENABLE_OMP)
  add_executable(tbsla_perf_omp tbsla_perf.cpp)
  target_link_libraries(tbsla_perf_omp PRIVATE tbsla_omp)
  install(TARGETS tbsla_perf_omp DESTINATION bin)

  add_executable(test_impl_csr_omp test_impl_csr.cpp)
  target_link_libraries(test_impl_csr_omp PUBLIC OpenMP::OpenMP_CXX)
  target_link_options(test_impl_csr_omp PUBLIC "${TBSLA_OpenMP_CXX_LIBRARIES}")
  install(TARGETS test_impl_csr_omp DESTINATION bin)
endif()
