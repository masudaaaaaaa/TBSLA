#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixSCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/hpx/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/vector.hpp>

void test_cqmat(int nr, int nc, int c, double q, unsigned int seed, int gr, int gc) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  tbsla::hpx_::Vector vcoo(gr, gc, 1);
  vcoo.init_single(nc);
  vcoo.wait();
  std::vector<double> v_data = vcoo.get_vectors().front().get_data().get().get_vect();

  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << "; q : " << q << "; s : " << seed << " ---- gr : " << gr << "; gc : " << gc << std::endl;
  tbsla::hpx_::MatrixCOO mcoo;
  mcoo.fill_cqmat(localities, nr, nc, c, q, seed, gr, gc);
  mcoo.wait();
  tbsla::hpx_::Vector r = mcoo.spmv(vcoo);
  std::vector<double> rcoo = r.get_vectors().front().get_data().get().get_vect();

  tbsla::hpx_::Vector v(gr, gc, gc);
  v.init_split(nc);
  v.wait();
  tbsla::hpx_::MatrixSCOO mscoo;
  mscoo.wait();
  mscoo.fill_cqmat(localities, nr, nc, c, q, seed, gr, gc);
  r = mscoo.spmv(v);
  std::vector<double> rscoo = r.get_vectors().front().get_data().get().get_vect();
  if(rscoo != rcoo) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v ", v_data);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcsr ", rscoo);
    std::cout << std::endl;
    throw "Result vector does not correspond to the expected results !";
  }

  tbsla::hpx_::MatrixCSR mcsr;
  mcsr.wait();
  mcsr.fill_cqmat(localities, nr, nc, c, q, seed, gr, gc);
  r = mcsr.spmv(v);
  std::vector<double> rcsr = r.get_vectors().front().get_data().get().get_vect();
  if(rcsr != rcoo) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v ", v_data);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcsr ", rcsr);
    std::cout << std::endl;
    throw "Result vector does not correspond to the expected results !";
  }

  tbsla::hpx_::MatrixELL mell;
  mell.fill_cqmat(localities, nr, nc, c, q, seed, gr, gc);
  mell.wait();
  r = mell.spmv(v);
  std::vector<double> rell = r.get_vectors().front().get_data().get().get_vect();
  if(rell != rcoo) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v ", v_data);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rell ", rell);
    std::cout << std::endl;
    throw "Result vector does not correspond to the expected results !";
  }

  tbsla::hpx_::MatrixDENSE mdense;
  mdense.fill_cqmat(localities, nr, nc, c, q, seed, gr, gc);
  mdense.wait();
  r = mdense.spmv(v);
  std::vector<double> rdense = r.get_vectors().front().get_data().get().get_vect();
  if(rdense != rcoo) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v ", v_data);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rdense ", rdense);
    std::cout << std::endl;
    throw "Result vector does not correspond to the expected results !";
  }
}

void test_mat(int nr, int nc, int c, int gr, int gc) {
  for(double s = 0; s < 2; s++) {
    for(double q = 0; q <= 1; q += 0.2) {
      test_cqmat(nr, nc, c, q, s, gr, gc);
    }
  }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  int t = 0;
  for(int i = 0; i <= 10; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 4; nt++) {
      test_mat(30, 30, 2 * i, nt, 1);
      test_mat(30, 30, 2 * i, 1, nt);
      test_mat(30, 30, 2 * i, nt, nt);

      test_mat(30, 20, 2 * i, nt, 1);
      test_mat(30, 20, 2 * i, 1, nt);
      test_mat(30, 20, 2 * i, nt, nt);

      test_mat(20, 30, 2 * i, nt, 1);
      test_mat(20, 30, 2 * i, 1, nt);
      test_mat(20, 30, 2 * i, nt, nt);
    }
  }
  for(int i = 0; i <= 10; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 4; nt++) {
      test_mat(100, 100, 2 * i, nt, 1);
      test_mat(100, 100, 2 * i, 1, nt);
      test_mat(100, 100, 2 * i, nt, nt);

      test_mat(80, 100, 2 * i, nt, 1);
      test_mat(80, 100, 2 * i, 1, nt);
      test_mat(80, 100, 2 * i, nt, nt);

      test_mat(100, 80, 2 * i, nt, 1);
      test_mat(100, 80, 2 * i, 1, nt);
      test_mat(100, 80, 2 * i, nt, nt);
    }
  }
  std::cout << "=== finished without error === " << std::endl;
  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  using namespace hpx::program_options;
  options_description desc_commandline;

  // Initialize and run HPX
  return hpx::init(desc_commandline, argc, argv);
}
