#include <tbsla/cpp/utils/range.hpp>

/*
 * compute the number of local values
 *
 */
size_t tbsla::utils::range::lnv(size_t size, int local_pos, int number_pos) {
  size_t n = size / number_pos;
  size_t mod = size % number_pos;
  if (local_pos < mod)
    n++;
  return n;
}

/*
 * compute the position of the first local value in the global array
 *
 */
size_t tbsla::utils::range::pflv(size_t size, int local_pos, int number_pos) {
  size_t mod = size % number_pos;
  size_t n = lnv(size, local_pos, number_pos) * local_pos;
  if (local_pos >= mod) {
    n += mod;
  }
  return n;
}

