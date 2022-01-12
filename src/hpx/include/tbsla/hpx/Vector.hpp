#ifndef TBSLA_HPX_Vector
#define TBSLA_HPX_Vector

#include <hpx/hpx.hpp>
#include <hpx/serialization/serialize.hpp>
#include <vector>
#include <numeric>

namespace tbsla { namespace hpx_ { namespace detail {

class Vector {
  public:
    std::vector<double> get_vect() const { return data_; }
    Vector(std::vector<double> v) : data_(v) {}
    Vector(std::size_t n, std::size_t s) : data_(n)
    {
      std::iota (std::begin(data_), std::end(data_), s);
    }
    Vector(std::size_t n) : data_(n)
    {
      std::iota (std::begin(data_), std::end(data_), 0);
    }
    Vector(Vector const& other) : data_(other.data_) {}
    Vector() : data_() {}

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& data_;
    }

   std::vector<double> data_;
};

}}}

namespace tbsla { namespace hpx_ { namespace server {

struct Vector : hpx::components::component_base<Vector>
{
    // construct new instances
    Vector()
      : data_()
    {
    }

    Vector(tbsla::hpx_::detail::Vector const& data)
      : data_(data)
    {
    }

    Vector(std::size_t n)
      : data_(n)
    {
    }

    Vector(std::size_t n, std::size_t s)
      : data_(n, s)
    {
    }

    // Access data.
    tbsla::hpx_::detail::Vector get_data() const
    {
        return data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(Vector, get_data);

private:
    tbsla::hpx_::detail::Vector data_;
};

}}}

HPX_REGISTER_ACTION_DECLARATION(tbsla::hpx_::server::Vector::get_data_action)

namespace tbsla { namespace hpx_ { namespace client {

struct Vector : hpx::components::client_base<Vector, tbsla::hpx_::server::Vector>
{
    typedef hpx::components::client_base<Vector, tbsla::hpx_::server::Vector> base_type;

    Vector() {}

    // Create new component on locality 'where' and initialize the held data
    Vector(hpx::id_type where, std::size_t n)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(where, n))
    {
    }

    Vector(hpx::id_type where, std::size_t n, std::size_t s)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(where, n, s))
    {
    }

    Vector(hpx::id_type where, tbsla::hpx_::detail::Vector const& data)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(hpx::colocated(where), data))
    {
    }

    Vector(hpx::id_type where)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(hpx::colocated(where)))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    Vector(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    Vector(hpx::future<tbsla::hpx_::client::Vector>&& c)
      : base_type(std::move(c))
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<tbsla::hpx_::detail::Vector> get_data() const
    {
        tbsla::hpx_::server::Vector::get_data_action act;
        return hpx::async(act, get_id());
    }

};

}}}

namespace tbsla { namespace hpx_ { namespace detail {
  static tbsla::hpx_::client::Vector reduce_part(tbsla::hpx_::client::Vector const& vc1, tbsla::hpx_::client::Vector const& vc2);
  static tbsla::hpx_::client::Vector gather_part(tbsla::hpx_::client::Vector const& vc1, tbsla::hpx_::client::Vector const& vc2);
  static tbsla::hpx_::client::Vector split_part(tbsla::hpx_::client::Vector const& v, int p, int m);
}}}

namespace tbsla { namespace hpx_ {

class Vector {
  public:
    Vector() {};

    /*
     * gr : number of blocks in the row dimension
     * gc : number of blocks in the column dimension
     * s : number of subvectors
    */
    Vector(std::size_t gr, std::size_t gc, std::size_t s);

    /*
     * gr : number of blocks in the row dimension
     * gc : number of blocks in the column dimension
    */
    Vector(std::vector<tbsla::hpx_::client::Vector> vectors, std::size_t gr, std::size_t gc);

    /*
     * nv : number of values in the global vector
    */
    void init_split(std::size_t nv);

    /*
     * nv : number of values in the global vector
    */
    void init_single(std::size_t nv);

    /*
     * s : number of subvectors
    */
    void split(std::size_t s);


    void wait();

    std::vector<tbsla::hpx_::client::Vector> get_vectors() {return vectors;}
    std::size_t get_gr() {return gr;}
    std::size_t get_gc() {return gc;}

  private:
    std::vector<tbsla::hpx_::client::Vector> vectors;
    std::vector<hpx::id_type> localities;
    std::size_t gr, gc;
};

}}

tbsla::hpx_::Vector reduce(tbsla::hpx_::Vector);
tbsla::hpx_::Vector gather(tbsla::hpx_::Vector);
tbsla::hpx_::Vector gather_reduce(tbsla::hpx_::Vector);
tbsla::hpx_::Vector add_vectors(tbsla::hpx_::Vector v1, tbsla::hpx_::Vector v2);


#endif
