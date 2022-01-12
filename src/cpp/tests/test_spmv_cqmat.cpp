#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/utils/array.hpp>

#include <iostream>
#include <numeric>

void print(tbsla::cpp::Matrix & m) {
  std::cout << "--------" << std::endl;
  std::cout << m << std::endl;
  std::cout << "--------" << std::endl;
}

void test_cqmat(int nr, int nc, int c, double q, double s) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << "; q : " << q << "; s : " << s << " ----  " << std::endl;
  double* v = new double[nc]();
  std::iota (v, v + nc, 0);

  tbsla::cpp::MatrixCOO mcoo;
  mcoo.fill_cqmat(nr, nc, c, q, s);
  double* rcoo = mcoo.spmv(v);

  tbsla::cpp::MatrixSCOO mscoo(mcoo);
  double* rscoo = mscoo.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rscoo, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mscoo);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rscoo", rscoo, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixSCOO mscoo2;
  mscoo2.fill_cqmat(nr, nc, c, q, s);
  double* rscoo2 = mscoo2.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rscoo2, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mscoo);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rscoo2", rscoo2, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixCSR mcsr(mcoo);
  double* rcsr = mcsr.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rcsr, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mcsr);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcsr", rcsr, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixCSR mcsr2;
  mcsr2.fill_cqmat(nr, nc, c, q, s);
  double* rcsr2 = mcsr2.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rcsr2, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mcsr2);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcsr2", rcsr2, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixELL mell(mcoo);
  double* rell = mell.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rell, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mell);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rell", rell, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixELL mell2;
  mell2.fill_cqmat(nr, nc, c, q, s);
  double* rell2 = mell2.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rell2, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mell);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rell2", rell2, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixDENSE mdense(mcoo);
  double* rdense = mdense.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rdense, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mdense);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rdense", rdense, nr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixDENSE mdense2;
  mdense2.fill_cqmat(nr, nc, c, q, s);
  double* rdense2 = mdense2.spmv(v);
  if(tbsla::utils::array::compare_arrays(rcoo, rdense2, nr)) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mdense);
    tbsla::utils::array::stream<double>(std::cout, "v", v, nc);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rcoo", rcoo, nr);
    std::cout << std::endl;
    tbsla::utils::array::stream<double>(std::cout, "rdense2", rdense2, nr);
    std::cout << std::endl;
    exit(1);
  }
  delete[] v;
  delete[] rcoo;
  delete[] rscoo;
  delete[] rscoo2;
  delete[] rcsr;
  delete[] rcsr2;
  delete[] rell;
  delete[] rell2;
  delete[] rdense;
  delete[] rdense2;
}

void test_mat(int nr, int nc, int c) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      test_cqmat(nr, nc, c, q, s);
    }
  }
}

int main(int argc, char** argv) {
  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(5, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(10, 5, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(20, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 20, 2 * i);
  }

  std::cout << "=== finished without error === " << std::endl;
}
