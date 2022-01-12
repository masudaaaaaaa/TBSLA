#include <tbsla/hpx/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/range.hpp>

typedef hpx::components::component<tbsla::hpx_::server::MatrixDENSE> MatrixDENSE_server_type;
HPX_REGISTER_COMPONENT(MatrixDENSE_server_type, MatrixDENSE_server);

typedef tbsla::hpx_::server::MatrixDENSE::get_data_action get_MatrixDENSE_data_action;
HPX_REGISTER_ACTION(get_MatrixDENSE_data_action);

std::size_t tbsla::hpx_::MatrixDENSE::get_n_col() {
  if(this->tiles.size() == 0)
    return 0;
  return tiles[0].get_data().get().get_n_col();
}

std::size_t tbsla::hpx_::MatrixDENSE::get_n_row() {
  if(this->tiles.size() == 0)
    return 0;
  return tiles[0].get_data().get().get_n_row();
}

void tbsla::hpx_::MatrixDENSE::fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t gr, std::size_t gc) {
  this->gr = gr;
  this->gc = gc;
  this->tiles.resize(gr * gc);
  this->localities = localities;
  std::size_t nl = this->localities.size();
  std::size_t nt = this->tiles.size();

  for (std::size_t pr = 0; pr < gr; pr++) {
    for (std::size_t pc = 0; pc < gc; pc++) {
      this->tiles[pr * gc + pc] = tbsla::hpx_::client::MatrixDENSE(this->localities[(pr * gc + pc) * nl / nt], nr, nc, cdiag, pr, pc, gr, gc);
    }
  }
}

void tbsla::hpx_::MatrixDENSE::fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t gr, std::size_t gc) {
  this->gr = gr;
  this->gc = gc;
  this->tiles.resize(gr * gc);
  this->localities = localities;
  std::size_t nl = this->localities.size();
  std::size_t nt = this->tiles.size();

  for (std::size_t pr = 0; pr < gr; pr++) {
    for (std::size_t pc = 0; pc < gc; pc++) {
      this->tiles[pr * gc + pc] = tbsla::hpx_::client::MatrixDENSE(this->localities[(pr * gc + pc) * nl / nt], nr, nc, c, q, seed, pr, pc, gr, gc);
    }
  }
}

static tbsla::hpx_::client::Vector tbsla::hpx_::detail::spmv_part(tbsla::hpx_::client::MatrixDENSE const& A_p, tbsla::hpx_::client::Vector const& v_p, tbsla::hpx_::client::Vector const& r_p) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::MatrixDENSE> A_data = A_p.get_data();
  hpx::shared_future<tbsla::hpx_::detail::Vector> v_data = v_p.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([r_p](tbsla::hpx_::detail::MatrixDENSE const& A, tbsla::hpx_::detail::Vector const& v) -> tbsla::hpx_::client::Vector {
      tbsla::hpx_::detail::Vector r(A.spmv(v.get_vect(), 0));
      return tbsla::hpx_::client::Vector(r_p.get_id(), r);
    }),
    A_data, v_data);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::spmv_part, spmv_dense_part_action);


tbsla::hpx_::Vector tbsla::hpx_::MatrixDENSE::spmv(tbsla::hpx_::Vector v) {
  std::vector<tbsla::hpx_::client::Vector> spmv_res;
  spmv_res.resize(this->tiles.size());
  for (std::size_t i = 0; i != this->tiles.size(); ++i) {
    spmv_res[i] = tbsla::hpx_::client::Vector(this->localities[i * this->localities.size() / this->tiles.size()]);
  }

  spmv_dense_part_action act_spmv;

  using hpx::dataflow;
  for (std::size_t i = 0; i < v.get_gr(); ++i) {
    for (std::size_t j = 0; j < v.get_gc(); ++j) {
      using hpx::util::placeholders::_1;
      using hpx::util::placeholders::_2;
      using hpx::util::placeholders::_3;
      auto Op = hpx::util::bind(act_spmv, this->localities[(i * gc + j) * this->localities.size() / this->tiles.size()], _1, _2, _3);
      spmv_res[i* gc + j] = dataflow(hpx::launch::async, Op, this->tiles[i * gc + j], v.get_vectors()[j], spmv_res[i * gc + j]);
    }
  }

  return gather_reduce(tbsla::hpx_::Vector(spmv_res, v.get_gr(), v.get_gc()));
}

tbsla::hpx_::Vector tbsla::hpx_::MatrixDENSE::a_axpx_(tbsla::hpx_::Vector v) {
  tbsla::hpx_::Vector r = this->spmv(v);
  r.split(this->gc);
  r = add_vectors(r, v);
  return this->spmv(r);
}

void tbsla::hpx_::MatrixDENSE::wait() {
  for (std::size_t i = 0; i < this->tiles.size(); ++i) {
    tiles[i].get_data().wait();
  }
}

