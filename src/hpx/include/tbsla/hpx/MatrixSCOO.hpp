#ifndef TBSLA_HPX_MatrixSCOO
#define TBSLA_HPX_MatrixSCOO

#include <tbsla/hpx/Matrix.hpp>
#include <tbsla/hpx/Vector.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>


namespace tbsla { namespace hpx_ { namespace detail {

class MatrixSCOO : public tbsla::cpp::MatrixSCOO, virtual tbsla::hpx_::detail::Matrix {
  public:
    using tbsla::cpp::MatrixSCOO::spmv;
    using tbsla::cpp::MatrixSCOO::fill_cdiag;
    using tbsla::cpp::MatrixSCOO::fill_cqmat;

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& row& col& values& gnnz& f_row& f_col& ln_row& ln_col;
    }
};

}}}


namespace tbsla { namespace hpx_ { namespace server {
struct HPX_COMPONENT_EXPORT MatrixSCOO : ::hpx::components::component_base<MatrixSCOO>
{
    // construct new instances
    MatrixSCOO() {}

    MatrixSCOO(tbsla::hpx_::detail::MatrixSCOO const& data)
      : data_(data)
    {
    }

    MatrixSCOO(std::size_t i, std::size_t n, std::string matrix_file)
      : data_()
    {
       std::ifstream is(matrix_file, std::ifstream::binary);
       data_.read(is, i, n);
       is.close();
    }

    MatrixSCOO(std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : data_()
    {
       data_.fill_cdiag(nr, nc, cdiag, pr, pc, NR, NC);
    }

    MatrixSCOO(std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : data_()
    {
       data_.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
    }

    // Access data.
    tbsla::hpx_::detail::MatrixSCOO get_data() const
    {
        return data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(MatrixSCOO, get_data);


private:
    tbsla::hpx_::detail::MatrixSCOO data_;
};


}}}

HPX_REGISTER_ACTION_DECLARATION(tbsla::hpx_::server::MatrixSCOO::get_data_action)

namespace tbsla { namespace hpx_ { namespace client {

struct MatrixSCOO : ::hpx::components::client_base<MatrixSCOO, tbsla::hpx_::server::MatrixSCOO>
{
    typedef ::hpx::components::client_base<MatrixSCOO, tbsla::hpx_::server::MatrixSCOO> base_type;

    MatrixSCOO() {}

    MatrixSCOO(hpx::id_type where, tbsla::hpx_::detail::MatrixSCOO const& data)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixSCOO>(hpx::colocated(where), data))
    {
    }

    MatrixSCOO(hpx::id_type where, std::size_t i, std::size_t n, std::string matrix_file)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixSCOO>(hpx::colocated(where), i, n, matrix_file))
    {
    }

    MatrixSCOO(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixSCOO>(hpx::colocated(where), nr, nc, cdiag, pr, pc, NR, NC))
    {
    }

    MatrixSCOO(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixSCOO>(hpx::colocated(where), nr, nc, c, q, seed, pr, pc, NR, NC))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    MatrixSCOO(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    MatrixSCOO(hpx::future<MatrixSCOO>&& c)
      : base_type(std::move(c))
    {
    }

    ::hpx::future<tbsla::hpx_::detail::MatrixSCOO> get_data() const
    {
        tbsla::hpx_::server::MatrixSCOO::get_data_action act;
        return ::hpx::async(act, get_id());
    }

};

}}}

namespace tbsla { namespace hpx_ { namespace detail {

static tbsla::hpx_::client::Vector spmv_part(tbsla::hpx_::client::MatrixSCOO const& A_p, tbsla::hpx_::client::Vector const& v_p, tbsla::hpx_::client::Vector const& r_p);

}}}

namespace tbsla { namespace hpx_ {

class MatrixSCOO : public tbsla::hpx_::Matrix {
  public:
    void fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t gr, std::size_t gc);
    void fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t gr, std::size_t gc);
    void wait();
    std::size_t get_n_col();
    std::size_t get_n_row();
    tbsla::hpx_::Vector spmv(tbsla::hpx_::Vector v);
    tbsla::hpx_::Vector a_axpx_(tbsla::hpx_::Vector v);
  private:
    std::vector<tbsla::hpx_::client::MatrixSCOO> tiles;
    std::vector<hpx::id_type> localities;
};

}}

#endif
