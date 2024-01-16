#pragma once

#define CMAKE_CXX_FLAGS "-Nclang -fPIC -Ofast -mcpu=native -funroll-loops -fno-builtin -march=armv8.2-a+sve"
#define CMAKE_CXX_FLAGS_DEBUG "-g"
#define CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG"
#define CMAKE_BUILD_TYPE "Release"
#define OpenMP_CXX_FLAGS "-fopenmp=libomp"
#define CMAKE_CXX_COMPILER "/opt/FJSVxtclanga/tcsds-1.2.38/bin/mpiFCC"
#define CMAKE_CXX_COMPILER_ID "Clang"
#define CMAKE_CXX_COMPILER_VERSION "7.1.0"
