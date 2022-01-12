add_executable(coo_to_csr coo_to_csr.cpp)
target_link_libraries(coo_to_csr PRIVATE tbsla)

add_executable(input_parser input_parser.cpp)
target_link_libraries(input_parser PRIVATE tbsla)
add_test(input_parser_test input_parser)

add_executable(test_spmv_cdiag test_spmv_cdiag.cpp)
target_link_libraries(test_spmv_cdiag PRIVATE tbsla)
add_test(test_spmv_cdiag_test test_spmv_cdiag)

add_executable(test_spmv_cqmat test_spmv_cqmat.cpp)
target_link_libraries(test_spmv_cqmat PRIVATE tbsla)
add_test(test_spmv_cqmat_test test_spmv_cqmat)

add_executable(test_a_axpx__cdiag test_a_axpx__cdiag.cpp)
target_link_libraries(test_a_axpx__cdiag PRIVATE tbsla)
add_test(test_a_axpx__cdiag_test test_a_axpx__cdiag)

add_executable(test_Ax_cdiag test_Ax_cdiag.cpp)
target_link_libraries(test_Ax_cdiag PRIVATE tbsla)
add_test(test_Ax_cdiag_test test_Ax_cdiag)

add_executable(test_AAxpAx_cdiag test_AAxpAx_cdiag.cpp)
target_link_libraries(test_AAxpAx_cdiag PRIVATE tbsla)
add_test(test_AAxpAx_cdiag_test test_AAxpAx_cdiag)

if(TBSLA_ENABLE_OMP)
  add_executable(test_spmv_cdiag_omp test_spmv_cdiag.cpp)
  target_link_libraries(test_spmv_cdiag_omp PRIVATE tbsla_omp)
  add_test(test_spmv_cdiag_test_omp test_spmv_cdiag_omp)

  add_executable(test_spmv_cqmat_omp test_spmv_cqmat.cpp)
  target_link_libraries(test_spmv_cqmat_omp PRIVATE tbsla_omp)
  add_test(test_spmv_cqmat_test_omp test_spmv_cqmat_omp)

  add_executable(test_a_axpx__cdiag_omp test_a_axpx__cdiag.cpp)
  target_link_libraries(test_a_axpx__cdiag_omp PRIVATE tbsla_omp)
  add_test(test_a_axpx__cdiag_test_omp test_a_axpx__cdiag_omp)

  add_executable(test_Ax_cdiag_omp test_Ax_cdiag.cpp)
  target_link_libraries(test_Ax_cdiag_omp PRIVATE tbsla_omp)
  add_test(test_Ax_cdiag_test_omp test_Ax_cdiag_omp)

  add_executable(test_AAxpAx_cdiag_omp test_AAxpAx_cdiag.cpp)
  target_link_libraries(test_AAxpAx_cdiag_omp PRIVATE tbsla_omp)
  add_test(test_AAxpAx_cdiag_test_omp test_AAxpAx_cdiag_omp)
endif()
