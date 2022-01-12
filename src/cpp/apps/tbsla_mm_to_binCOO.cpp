#include <tbsla/cpp/MatrixCOO.hpp>

#include <fstream>


int main(int argc, char** argv) {

  if(argc == 3) {
    tbsla::cpp::MatrixCOO m;
    m.readMM(std::string(argv[1]));
    std::ofstream os(std::string(argv[2]), std::ofstream::binary);
    m.write(os);
    os.close();

    std::cout << "== read m2 ==" << std::endl;
    std::ifstream is(std::string(argv[2]), std::ifstream::binary);
    tbsla::cpp::MatrixCOO m2;
    m2.read(is);
    is.close();
  }
}
