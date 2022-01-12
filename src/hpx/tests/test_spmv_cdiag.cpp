#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixSCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/hpx/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/vector.hpp>

void test_mat_split_vector(tbsla::hpx_::Matrix & m, int nr, int nc, int c, int gr, int gc) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  tbsla::hpx_::Vector vcoo(gr, gc, 1);
  vcoo.init_single(nc);
  vcoo.wait();
  std::vector<double> v_data = vcoo.get_vectors().front().get_data().get().get_vect();
  tbsla::hpx_::Vector v(gr, gc, gc);
  v.init_split(nc);
  v.wait();

  m.fill_cdiag(localities, nr, nc, c, gr, gc);
  m.wait();
  tbsla::hpx_::Vector r = m.spmv(v);
  std::vector<double> r_data = r.get_vectors().front().get_data().get().get_vect();
  int res = tbsla::utils::vector::test_spmv_cdiag(nr, nc, c, v_data, r_data, false);
  std::cout << "return : " << res << std::endl;
  if(res) {
    tbsla::utils::vector::test_spmv_cdiag(nr, nc, c, v_data, r_data, true);
    throw "Result vector does not correspond to the expected results !";
  }
}

void test_mat(tbsla::hpx_::Matrix & m, int nr, int nc, int c, int gr, int gc) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  tbsla::hpx_::Vector vcoo(gr, gc, 1);
  vcoo.init_single(nc);
  vcoo.wait();
  std::vector<double> v_data = vcoo.get_vectors().front().get_data().get().get_vect();

  m.fill_cdiag(localities, nr, nc, c, gr, gc);
  m.wait();
  tbsla::hpx_::Vector r = m.spmv(vcoo);
  std::vector<double> r_data = r.get_vectors().front().get_data().get().get_vect();
  int res = tbsla::utils::vector::test_spmv_cdiag(nr, nc, c, v_data, r_data, false);
  std::cout << "return : " << res << std::endl;
  if(res) {
    tbsla::utils::vector::test_spmv_cdiag(nr, nc, c, v_data, r_data, true);
    throw "Result vector does not correspond to the expected results !";
  }
}


void test_cdiag(int nr, int nc, int c, int gr, int gc) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << " ---- gr : " << gr << "; gc : " << gc << std::endl;
  tbsla::hpx_::MatrixCOO mcoo;
  test_mat(mcoo, nr, nc, c, gr, gc);

  tbsla::hpx_::MatrixSCOO mscoo;
  test_mat_split_vector(mscoo, nr, nc, c, gr, gc);

  tbsla::hpx_::MatrixCSR mcsr;
  test_mat_split_vector(mcsr, nr, nc, c, gr, gc);

  tbsla::hpx_::MatrixELL mell;
  test_mat_split_vector(mell, nr, nc, c, gr, gc);

  tbsla::hpx_::MatrixDENSE mdense;
  test_mat_split_vector(mdense, nr, nc, c, gr, gc);
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  int t = 0;
  for(int i = 0; i <= 10; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(30, 30, 2 * i, nt, 1);
      test_cdiag(30, 30, 2 * i, 1, nt);
      test_cdiag(30, 30, 2 * i, nt, nt);

      test_cdiag(20, 30, 2 * i, nt, 1);
      test_cdiag(20, 30, 2 * i, 1, nt);
      test_cdiag(20, 30, 2 * i, nt, nt);

      test_cdiag(30, 20, 2 * i, nt, 1);
      test_cdiag(30, 20, 2 * i, 1, nt);
      test_cdiag(30, 20, 2 * i, nt, nt);
    }
  }
  for(int i = 0; i <= 10; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(100, 100, 2 * i, nt, 1);
      test_cdiag(100, 100, 2 * i, 1, nt);
      test_cdiag(100, 100, 2 * i, nt, nt);

      test_cdiag(80, 100, 2 * i, nt, 1);
      test_cdiag(80, 100, 2 * i, 1, nt);
      test_cdiag(80, 100, 2 * i, nt, nt);

      test_cdiag(100, 80, 2 * i, nt, 1);
      test_cdiag(100, 80, 2 * i, 1, nt);
      test_cdiag(100, 80, 2 * i, nt, nt);
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
