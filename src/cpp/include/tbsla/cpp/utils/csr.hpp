#ifndef TBSLA_CPP_UTILS_CSR
#define TBSLA_CPP_UTILS_CSR

namespace tbsla { namespace cpp { namespace utils { namespace csr {

template <typename T>
T* applyPermutation(const int* order, const T* t, int size) {
  T* st = new T[size];
  for(int i = 0; i < size; i++) {
    st[i] = t[order[i]];
  }
  return st;
}

bool compare_row(const int* row, const int* col, unsigned i, unsigned j);

}}}}

#endif
