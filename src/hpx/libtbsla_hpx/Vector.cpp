#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/hpx/Vector.hpp>
#include <hpx/hpx.hpp>

typedef hpx::components::component<tbsla::hpx_::server::Vector> Vector_server_type;
HPX_REGISTER_COMPONENT(Vector_server_type, Vector_server);

typedef tbsla::hpx_::server::Vector::get_data_action get_Vector_data_action;
HPX_REGISTER_ACTION(get_Vector_data_action);

static tbsla::hpx_::client::Vector tbsla::hpx_::detail::reduce_part(tbsla::hpx_::client::Vector const& vc1, tbsla::hpx_::client::Vector const& vc2) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::Vector> vd1 = vc1.get_data();
  hpx::shared_future<tbsla::hpx_::detail::Vector> vd2 = vc2.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([vc1](tbsla::hpx_::detail::Vector const& v1, tbsla::hpx_::detail::Vector const& v2) -> tbsla::hpx_::client::Vector {
      std::vector<double> vv1 = v1.get_vect();
      std::vector<double> vv2 = v2.get_vect();
      std::vector<double> rv(vv1.size());
      if(vv1.size() == vv2.size()) {
         for(std::size_t i = 0; i < vv1.size(); i++)
           rv[i] = vv1[i] + vv2[i];
      }
      tbsla::hpx_::detail::Vector r(rv);
      return tbsla::hpx_::client::Vector(vc1.get_id(), r);
    }),
    vd1, vd2);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::reduce_part, reduce_part_action);


static tbsla::hpx_::client::Vector tbsla::hpx_::detail::gather_part(tbsla::hpx_::client::Vector const& vc1, tbsla::hpx_::client::Vector const& vc2) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::Vector> vd1 = vc1.get_data();
  hpx::shared_future<tbsla::hpx_::detail::Vector> vd2 = vc2.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([vc1](tbsla::hpx_::detail::Vector const& v1, tbsla::hpx_::detail::Vector const& v2) -> tbsla::hpx_::client::Vector {
      std::vector<double> vv1 = v1.get_vect();
      std::vector<double> vv2 = v2.get_vect();
      std::vector<double> rv(vv1.size() + vv2.size());
      for(std::size_t i = 0; i < vv1.size(); i++)
        rv[i] = vv1[i];
      for(std::size_t i = 0; i < vv2.size(); i++)
        rv[i + vv1.size()] = vv2[i];
      tbsla::hpx_::detail::Vector r(rv);
      return tbsla::hpx_::client::Vector(vc1.get_id(), r);
    }),
    vd1, vd2);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::gather_part, gather_part_action);

static tbsla::hpx_::client::Vector tbsla::hpx_::detail::split_part(tbsla::hpx_::client::Vector const& vc, int p, int m) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::Vector> vd = vc.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([vc, p, m](tbsla::hpx_::detail::Vector const& v) -> tbsla::hpx_::client::Vector {
      std::vector<double> vv = v.get_vect();
      std::size_t n = tbsla::utils::range::lnv(vv.size(), p, m);
      std::size_t s = tbsla::utils::range::pflv(vv.size(), p, m);

      std::vector<double> rv(n);
      for(std::size_t i = 0; i < n; i++)
        rv[i] = vv[i + s];
      tbsla::hpx_::detail::Vector r(rv);
      return tbsla::hpx_::client::Vector(vc.get_id(), r);
    }),
    vd);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::split_part, split_part_action);


tbsla::hpx_::Vector add_vectors(tbsla::hpx_::Vector v1, tbsla::hpx_::Vector v2) {
  using hpx::util::placeholders::_1;
  using hpx::util::placeholders::_2;
  using hpx::dataflow;

  reduce_part_action reduce_act;
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();

  std::vector<tbsla::hpx_::client::Vector> v1v = v1.get_vectors();
  std::vector<tbsla::hpx_::client::Vector> v2v = v2.get_vectors();
  std::vector<tbsla::hpx_::client::Vector> rv(v1v.size());

  for(std::size_t i = 0; i < v1v.size(); i++) {
    auto Op = hpx::util::bind(reduce_act, localities[i * nl / v1v.size()], _1, _2);
    rv[i] = dataflow(hpx::launch::async, Op, v1v[i] ,v2v[i]);
  }

  tbsla::hpx_::Vector r(rv, v1.get_gr(), v1.get_gc());
  return r;
}

std::vector<tbsla::hpx_::client::Vector> reduce_recursion(std::vector<tbsla::hpx_::client::Vector> vcs) {
  if(vcs.size() < 1) {
    throw "error in recursive reduction";
  } else if(vcs.size() == 1) {
    return vcs;
  } else {
    std::vector<tbsla::hpx_::client::Vector> new_vcs;
    int mod = vcs.size() % 2;
    size_t div2 = vcs.size() / 2;
    new_vcs.reserve(mod + div2);

    reduce_part_action reduce_act;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();

    using hpx::dataflow;
    for(std::size_t i = 0; i < div2; i++) {
      using hpx::util::placeholders::_1;
      using hpx::util::placeholders::_2;
      auto Op = hpx::util::bind(reduce_act, localities[i * nl / div2], _1, _2);
      new_vcs.push_back(dataflow(hpx::launch::async, Op, vcs[2 * i], vcs[2 * i + 1]));
    }
    if(mod == 1) {
      new_vcs.push_back(vcs.back());
    }
    return reduce_recursion(new_vcs);
  }
}

tbsla::hpx_::Vector reduce(tbsla::hpx_::Vector vcs) {
  return tbsla::hpx_::Vector(reduce_recursion(vcs.get_vectors()), vcs.get_gr(), vcs.get_gc());
}

std::vector<tbsla::hpx_::client::Vector> gather_recursion(std::vector<tbsla::hpx_::client::Vector> vcs) {
  if(vcs.size() < 1) {
    throw "error in recursive gathering";
  } else if(vcs.size() == 1) {
    return vcs;
  } else {
    std::vector<tbsla::hpx_::client::Vector> new_vcs;
    int mod = vcs.size() % 2;
    size_t div2 = vcs.size() / 2;
    new_vcs.reserve(mod + div2);

    gather_part_action gather_act;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();

    using hpx::dataflow;
    for(std::size_t i = 0; i < div2; i++) {
      using hpx::util::placeholders::_1;
      using hpx::util::placeholders::_2;
      auto Op = hpx::util::bind(gather_act, localities[i * nl / div2], _1, _2);
      new_vcs.push_back(dataflow(hpx::launch::async, Op, vcs[2 * i], vcs[2 * i + 1]));
    }
    if(mod == 1) {
      new_vcs.push_back(vcs.back());
    }
    return gather_recursion(new_vcs);
  }
}

tbsla::hpx_::Vector gather(tbsla::hpx_::Vector vcs) {
  return tbsla::hpx_::Vector(gather_recursion(vcs.get_vectors()), vcs.get_gr(), vcs.get_gc());
}

tbsla::hpx_::Vector gather_reduce(tbsla::hpx_::Vector vcs) {
  if(vcs.get_gr() == 1) {
    return reduce(vcs);
  } else if(vcs.get_gc() == 1) {
    return gather(vcs);
  } else {
    std::vector<tbsla::hpx_::client::Vector> togather(vcs.get_gr());
    std::vector<tbsla::hpx_::client::Vector> curr_vectors = vcs.get_vectors();

    for (std::size_t i = 0; i < vcs.get_gr(); ++i) {
      std::vector<tbsla::hpx_::client::Vector> toreduce(vcs.get_gc());
      for (std::size_t j = 0; j < vcs.get_gc(); ++j) {
        toreduce[j] = curr_vectors[i * vcs.get_gc() + j];
      }
      togather[i] = reduce_recursion(toreduce).front();
    }
    return tbsla::hpx_::Vector(gather_recursion(togather), vcs.get_gr(), vcs.get_gc());
  }
}

void tbsla::hpx_::Vector::init_split(std::size_t nv) {
  this->vectors.resize(this->gc);
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = this->localities.size();
  for(std::size_t i = 0; i < this->gc; i++) {
    std::size_t n = tbsla::utils::range::lnv(nv, i, this->gc);
    std::size_t s = tbsla::utils::range::pflv(nv, i, this->gc);
    this->vectors[i] = tbsla::hpx_::client::Vector(localities[i * nl / this->gc], n, s);
  }
}

void tbsla::hpx_::Vector::init_single(std::size_t nv) {
  this->vectors.resize(1);
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  this->vectors[0] = tbsla::hpx_::client::Vector(localities[0], nv, 0);
}

tbsla::hpx_::Vector::Vector(std::size_t gr, std::size_t gc, std::size_t s) {
  this->vectors.resize(s);
  this->gc = gc;
  this->gr = gr;
}

void tbsla::hpx_::Vector::split(std::size_t s) {
  if(s == 1)
    return;
  tbsla::hpx_::client::Vector tosplit = this->vectors.front();
  this->vectors.resize(s);
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = this->localities.size();

  using hpx::util::placeholders::_1;
  using hpx::util::placeholders::_2;
  using hpx::util::placeholders::_3;
  using hpx::dataflow;
  split_part_action split_act;

  for(std::size_t i = 0; i < s; i++) {
    auto Op = hpx::util::bind(split_act, localities[i * nl / s], _1, _2, _3);
    this->vectors[i] = dataflow(hpx::launch::async, Op, tosplit, i, s);
  }
}

tbsla::hpx_::Vector::Vector(std::vector<tbsla::hpx_::client::Vector> vectors, std::size_t gr, std::size_t gc) {
  this->vectors = vectors;
  this->gc = gc;
  this->gr = gr;
}

void tbsla::hpx_::Vector::wait() {
  for (std::size_t i = 0; i < this->vectors.size(); ++i) {
    this->vectors[i].get_data().wait();
  }
}
