add_executable(hpx_test_spmv_cdiag test_spmv_cdiag.cpp)
target_link_libraries(hpx_test_spmv_cdiag PUBLIC tbsla_hpx)
target_link_libraries(hpx_test_spmv_cdiag PUBLIC tbsla)
add_test(hpx_test_spmv_cdiag_test hpx_test_spmv_cdiag)

add_executable(hpx_test_spmv_cqmat test_spmv_cqmat.cpp)
target_link_libraries(hpx_test_spmv_cqmat PUBLIC tbsla_hpx)
target_link_libraries(hpx_test_spmv_cqmat PUBLIC tbsla)
add_test(hpx_test_spmv_cqmat_test hpx_test_spmv_cqmat)

add_executable(hpx_test_a_axpx__cdiag test_a_axpx__cdiag.cpp)
target_link_libraries(hpx_test_a_axpx__cdiag PUBLIC tbsla_hpx)
target_link_libraries(hpx_test_a_axpx__cdiag PUBLIC tbsla)
add_test(hpx_test_a_axpx__cdiag_test hpx_test_a_axpx__cdiag)

