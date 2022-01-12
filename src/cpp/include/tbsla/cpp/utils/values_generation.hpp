#ifndef TBSLA_CPP_VALUES_GENERATION
#define TBSLA_CPP_VALUES_GENERATION

#include <tuple>
#include <random>

namespace tbsla { namespace utils { namespace values_generation {
  std::tuple<std::size_t, std::size_t, double, std::size_t> cdiag_value(std::size_t i, std::size_t nv, std::size_t nr, std::size_t nc, std::size_t cdiag);
  std::tuple<std::size_t, std::size_t, double, std::size_t> cqmat_value(std::size_t i, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed_mult);
  double* cqmat_sum_columns(std::size_t nr, std::size_t nc, std::size_t c_, double q, unsigned int seed_mult);
  //int* random_columns(std::size_t n_vals, std::size_t range, std::uniform_real_distribution<double> distr_ind, std::default_random_engine generator);
  int* random_columns(std::size_t i, std::size_t n_vals, std::size_t nc, unsigned seed_mult);
  int* fix_list(int* list, std::size_t n_vals, std::size_t nc);
}}}

#endif
