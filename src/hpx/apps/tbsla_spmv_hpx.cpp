#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>

#include <random>
#include <map>
#include <algorithm>

int hpx_main(hpx::program_options::variables_map& vm)
{
  std::uint64_t N = vm["N"].as<std::uint64_t>();
  std::string matrix_input = vm["matrix_input"].as<std::string>();
  std::string format = vm["format"].as<std::string>();

  if(matrix_input == "") {
    std::cerr << "A matrix file has to be given with the parameter --matrix_input file" << std::endl;
    return hpx::finalize();
  }
  if(format == "") {
    std::cerr << "A file format has to be given with the parameter --format format" << std::endl;
    return hpx::finalize();
  }

  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();    // Number of localities

  if (N < nl)
  {
    std::cout << "The number of tiles should not be smaller than "
                 "the number of localities"
              << std::endl;
    return hpx::finalize();
  }

  tbsla::hpx_::Matrix *m;

  if(format == "COO" | format == "coo") {
    m = new tbsla::hpx_::MatrixCOO();
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::hpx_::MatrixCSR();
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::hpx_::MatrixELL();
  } else {
    std::cerr << format << " unrecognized!" << std::endl;
    return hpx::finalize();
  }

  std::uint64_t t_read_start = hpx::util::high_resolution_clock::now();
  m->read(localities, matrix_input, N);
  std::uint64_t t_read_end = hpx::util::high_resolution_clock::now();

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
  std::vector<double> vec(m->get_n_col());
  std::generate(begin(vec), end(vec), gen);
  tbsla::hpx_::detail::Vector vdata(vec);
  tbsla::hpx_::client::Vector v(localities[0], vdata);

  std::uint64_t t_spmv_start = hpx::util::high_resolution_clock::now();
  tbsla::hpx_::client::Vector r = m->spmv(v);
  r.get_data().wait();
  std::uint64_t t_spmv_end = hpx::util::high_resolution_clock::now();

  std::map<std::string, std::string> outmap;
  outmap["test"] = "spmv";
  outmap["matrix_format"] = format;
  outmap["n_row"] = std::to_string(m->get_n_row());
  outmap["n_col"] = std::to_string(m->get_n_col());
  outmap["time_read_m"] = std::to_string((t_read_end - t_read_start) / 1e9);
  outmap["time_spmv"] = std::to_string((t_spmv_end - t_spmv_start) / 1e9);
  outmap["lang"] = "HPX";
  outmap["matrix_input"] = matrix_input;

  std::map<std::string, std::string>::iterator it=outmap.begin();
  std::cout << "{\"" << it->first << "\":\"" << it->second << "\"";
  it++;
  for (; it != outmap.end(); ++it) {
    std::cout << ",\"" << it->first << "\":\"" << it->second << "\"";
  }
  std::cout << "}\n";

  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  using namespace hpx::program_options;

  options_description desc_commandline;
  desc_commandline.add_options()(
    "print-input", "print matrix and vector")("N",
    value<std::uint64_t>()->default_value(10),
    "Dimension of the submatrices")("matrix_input",
    value<std::string>()->default_value(""),
    "file containing the matrix (default : empty)")
    ("format", value<std::string>()->default_value(""),
    "storage format of the matrix (default : empty)");

  // Initialize and run HPX
  return hpx::init(desc_commandline, argc, argv);
}
