#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>

#include <fstream>


int main(int argc, char** argv) {
  InputParser parser(argc, argv);

  std::string input = parser.get_opt("--input");
  if(input == "") {
    std::cerr << "An input file containing COO matrix has to be given with the parameter --input input" << std::endl;
    exit(1);
  }

  std::string output = parser.get_opt("--output");
  if(output == "") {
    std::cerr << "An output file has to be given with the parameter --output output" << std::endl;
    exit(1);
  }

  std::string format = parser.get_opt("--format");
  if(format == "") {
    std::cerr << "A matrix format for the output has to be given with the parameter --format format" << std::endl;
    exit(1);
  }

  std::ifstream is(input, std::ifstream::binary);
  tbsla::cpp::MatrixCOO mcoo;
  mcoo.read(is);
  is.close();

  tbsla::cpp::Matrix *m;

  if(format == "SCOO" | format == "scoo") {
    m = new tbsla::cpp::MatrixSCOO(mcoo);
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::cpp::MatrixCSR(mcoo);
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::cpp::MatrixELL(mcoo);
  } else if(format == "DENSE" | format == "dense") {
    m = new tbsla::cpp::MatrixDENSE(mcoo);
  } else {
    std::cerr << format << " unrecognized!" << std::endl;
    exit(1);
  }

  std::ofstream os(output, std::ofstream::binary);
  m->write(os);
  os.close();
}
