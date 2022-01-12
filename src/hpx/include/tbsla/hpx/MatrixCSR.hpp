#ifndef TBSLA_HPX_MatrixCSR
#define TBSLA_HPX_MatrixCSR

#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/hpx/Matrix.hpp>
#include <tbsla/hpx/Vector.hpp>

#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

namespace tbsla { namespace hpx_ { namespace detail {

class MatrixCSR : public tbsla::cpp::MatrixCSR, public tbsla::hpx_::detail::Matrix {
  public:
    using tbsla::cpp::MatrixCSR::spmv;
    using tbsla::cpp::MatrixCSR::fill_cdiag;
    using tbsla::cpp::MatrixCSR::fill_cqmat;

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& rowptr& colidx& values& gnnz& f_row& f_col& ln_row& ln_col;
    }
};

}}}


namespace tbsla { namespace hpx_ { namespace server {
struct HPX_COMPONENT_EXPORT MatrixCSR : ::hpx::components::component_base<MatrixCSR>
{
    // construct new instances
    MatrixCSR() {}

    MatrixCSR(tbsla::hpx_::detail::MatrixCSR const& data)
      : data_(data)
    {
    }

    MatrixCSR(std::size_t i, std::size_t n, std::string matrix_file)
      : data_()
    {
       std::ifstream is(matrix_file, std::ifstream::binary);
       data_.read(is, i, n);
       is.close();
    }

    MatrixCSR(std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : data_()
    {
       data_.fill_cdiag(nr, nc, cdiag, pr, pc, NR, NC);
    }

    MatrixCSR(std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : data_()
    {
       data_.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
    }

    // Access data.
    tbsla::hpx_::detail::MatrixCSR get_data() const
    {
        return data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(MatrixCSR, get_data);


private:
    tbsla::hpx_::detail::MatrixCSR data_;
};


}}}

HPX_REGISTER_ACTION_DECLARATION(tbsla::hpx_::server::MatrixCSR::get_data_action)

namespace tbsla { namespace hpx_ { namespace client {

struct MatrixCSR : ::hpx::components::client_base<MatrixCSR, tbsla::hpx_::server::MatrixCSR>
{
    typedef ::hpx::components::client_base<MatrixCSR, tbsla::hpx_::server::MatrixCSR> base_type;

    MatrixCSR() {}

    MatrixCSR(hpx::id_type where, tbsla::hpx_::detail::MatrixCSR const& data)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixCSR>(hpx::colocated(where), data))
    {
    }

    MatrixCSR(hpx::id_type where, std::size_t i, std::size_t n, std::string matrix_file)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixCSR>(hpx::colocated(where), i, n, matrix_file))
    {
    }

    MatrixCSR(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixCSR>(hpx::colocated(where), nr, nc, cdiag, pr, pc, NR, NC))
    {
    }

    MatrixCSR(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pr, std::size_t pc, std::size_t NR, std::size_t NC)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixCSR>(hpx::colocated(where), nr, nc, c, q, seed, pr, pc, NR, NC))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    MatrixCSR(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    MatrixCSR(hpx::future<MatrixCSR>&& c)
      : base_type(std::move(c))
    {
    }

    ::hpx::future<tbsla::hpx_::detail::MatrixCSR> get_data() const
    {
        tbsla::hpx_::server::MatrixCSR::get_data_action act;
        return ::hpx::async(act, get_id());
    }

};

}}}

namespace tbsla { namespace hpx_ { namespace detail {

static tbsla::hpx_::client::Vector spmv_part(tbsla::hpx_::client::MatrixCSR const& A_p, tbsla::hpx_::client::Vector const& v_p, tbsla::hpx_::client::Vector const& r_p);

}}}

namespace tbsla { namespace hpx_ {

class MatrixCSR : public tbsla::hpx_::Matrix {
  public:
    void fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t gr, std::size_t gc);
    void fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t gr, std::size_t gc);
    void wait();
    std::size_t get_n_col();
    std::size_t get_n_row();
    tbsla::hpx_::Vector spmv(tbsla::hpx_::Vector v);
    tbsla::hpx_::Vector a_axpx_(tbsla::hpx_::Vector v);
  private:
    std::vector<tbsla::hpx_::client::MatrixCSR> tiles;
    std::vector<hpx::id_type> localities;
};

}}

#endif
