#include <tbsla/cpp/utils/values_generation.hpp>
#include <cstdlib>

std::tuple<std::size_t, std::size_t, double, std::size_t> tbsla::utils::values_generation::cdiag_value(std::size_t i, std::size_t nv, std::size_t nr, std::size_t nc, std::size_t cdiag) {
  if(cdiag == 0) {
    return std::make_tuple(i, i, 1, 10);
  }
  if(i < std::max(std::min((long int)nc - (long int)cdiag, (long int)cdiag), (long int)0)) {
    return std::make_tuple(i, i + cdiag, 1, 30);
  } else if (i < std::max((long int)cdiag + 2 * ((long int)nc - 2 * (long int)cdiag), (long int)0)) {
    long int it = (i + cdiag) / 2;
    if(i % 2 == 0) {
      return std::make_tuple(it, it - cdiag, 1, 31);
    } else {
      return std::make_tuple(it, it + cdiag, 1, 32);
    }
  } else {
    long int it = i - ((long int)nc - 2 * (long int)cdiag);
    if(cdiag > nc) {
      it -= cdiag - nc;
    }
    return std::make_tuple(it, it - cdiag, 1, 33);
  }
}

std::tuple<std::size_t, std::size_t, double, std::size_t> tbsla::utils::values_generation::cqmat_value(std::size_t i, std::size_t nr, std::size_t nc, std::size_t c_, double q, unsigned int seed_mult) {
  unsigned int seedp = i;
  if(seed_mult > 0) {
    seedp = seed_mult * i;
  }
  std::size_t c = std::min(nc, c_);
  if(i < c * std::min(nc - c + 1, nr)) {
    return std::make_tuple(i / c, ((double)rand_r(&seedp) / RAND_MAX) < q ? rand_r(&seedp) % nc : i / c + i % c, 1, 30);
  } else {
    std::size_t n_full_rows = std::min(nc - c + 1, nr);
    std::size_t it = i - c * std::min(nc - c + 1, nr);
    std::size_t curr_row = 1;
    while(curr_row < nr - n_full_rows && it >= c - curr_row) {
      it -= (c - curr_row);
      curr_row++;
    }
    curr_row += n_full_rows - 1;
    return std::make_tuple(curr_row, (double)(rand_r(&seedp) / RAND_MAX) < q ? rand_r(&seedp) % nc : curr_row + it, 1, 31);
  }
}

double* tbsla::utils::values_generation::cqmat_sum_columns(std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned seed_mult) {
  std::size_t gnv = 0;
  for(std::size_t i = 0; i < std::min(nc - std::min(c, nc) + 1, nr); i++) {
    gnv += std::min(c, nc);
  }
  for(std::size_t i = 0; i < std::min(nr, nc) - std::min(nc - std::min(c, nc) + 1, nr); i++) {
    gnv += std::min(c, nc) - i - 1;
  }

  double* sum = new double[nc];
  for (int i = 0; i < nc; i++) {
    sum[i] = 0;
  }
  for(std::size_t i = 0; i < gnv; i++) {
    auto tuple = tbsla::utils::values_generation::cqmat_value(i, nr, nc, c, q, seed_mult);
    sum[std::get<1>(tuple)] += std::get<2>(tuple);
  }
  return sum;
}
