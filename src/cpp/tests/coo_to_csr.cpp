#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {

  std::vector<double> values{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
  std::vector<int> row{ 0, 0, 0, 1, 2, 3, 3, 4, 4, 4, 6, 6, 5, 5 };
  std::vector<int> col{ 0, 1, 4, 2, 1, 1, 3, 0, 2, 4, 4, 0, 5, 3 };

  std::vector<double> x{ 1, 2, 3, 4, 5 };

  tbsla::cpp::MatrixCOO m(7, 7, values.data(), row.data(), col.data());
  std::cout << m << std::endl;

  tbsla::cpp::MatrixCSR m2(m);

  std::cout << m2 << std::endl;

}
