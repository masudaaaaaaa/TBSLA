#ifndef TBSLA_HPX_Matrix
#define TBSLA_HPX_Matrix

#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/hpx/Vector.hpp>

#include <hpx/hpx.hpp>

#include <fstream>
#include <vector>

namespace tbsla { namespace hpx_ { namespace detail {

class Matrix : public virtual tbsla::cpp::Matrix {
  public:
    int const get_gnnz() {return gnnz;};

  protected:
    int gnnz;
};

}}}

namespace tbsla { namespace hpx_ {

class Matrix {
  public:
    virtual void fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t gr, std::size_t gc) = 0;
    virtual void fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t gr, std::size_t gc) = 0;
    virtual void wait() = 0;
    virtual std::size_t get_n_col() = 0;
    virtual std::size_t get_n_row() = 0;
    virtual tbsla::hpx_::Vector spmv(tbsla::hpx_::Vector v) = 0;
    virtual tbsla::hpx_::Vector a_axpx_(tbsla::hpx_::Vector v) = 0;
  protected:
    int gr, gc;
};

}}
#endif
