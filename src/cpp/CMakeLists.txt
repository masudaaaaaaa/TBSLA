add_subdirectory(matrix)
add_subdirectory(utils)
add_subdirectory(apps)
add_subdirectory(tests)

if(TBSLA_C_INTERFACE)
  add_subdirectory(interface/c/matrix)
endif()
