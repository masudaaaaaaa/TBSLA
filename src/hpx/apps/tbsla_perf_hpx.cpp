#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixSCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/hpx/MatrixDENSE.hpp>

#include <random>
#include <map>
#include <algorithm>

int hpx_main(hpx::program_options::variables_map& vm)
{
  std::string format = vm["format"].as<std::string>();
  std::string matrix = vm["matrix"].as<std::string>();
  std::string op = vm["op"].as<std::string>();
  std::uint64_t GR = vm["GR"].as<std::uint64_t>();
  std::uint64_t GC = vm["GC"].as<std::uint64_t>();
  std::uint64_t NR = vm["NR"].as<std::uint64_t>();
  std::uint64_t NC = vm["NC"].as<std::uint64_t>();
  std::uint64_t C = vm["C"].as<std::uint64_t>();
  double Q = vm["Q"].as<double>();
  std::uint64_t S = vm["S"].as<std::uint64_t>();

  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();    // Number of localities

  if (GR * GC < nl)
  {
    std::cout << "The number of tiles should not be smaller than "
                 "the number of localities"
              << std::endl;
    return hpx::finalize();
  }

  if(matrix != "cdiag" && matrix != "cqmat") {
    std::cerr << "No matrix type has been given by --matrix matrix" << std::endl;
    return hpx::finalize();
  }
  if(op == "") {
    std::cerr << "An operation (spmv, a_axpx) has to be given with the parameter --op op" << std::endl;
    return hpx::finalize();
  }
  if(op != "spmv" && op != "a_axpx") {
    std::cerr << "OP : " << op << " unrecognized!" << std::endl;
    return hpx::finalize();
  }

  tbsla::hpx_::Matrix *m;

  if(format == "COO" | format == "coo") {
    m = new tbsla::hpx_::MatrixCOO();
  } else if(format == "SCOO" | format == "scoo") {
    m = new tbsla::hpx_::MatrixSCOO();
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::hpx_::MatrixCSR();
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::hpx_::MatrixELL();
  } else if(format == "DENSE" | format == "dense") {
    m = new tbsla::hpx_::MatrixDENSE();
  } else {
    std::cerr << format << " unrecognized!" << std::endl;
    return hpx::finalize();
  }

  std::uint64_t t_app_start = hpx::util::high_resolution_clock::now();
  if(matrix == "cdiag") {
    m->fill_cdiag(localities, NR, NC, C, GR, GC);
  } else if(matrix == "cqmat") {
    m->fill_cqmat(localities, NR, NC, C, Q, S, GR, GC);
  }
  m->wait();

  tbsla::hpx_::Vector v(GR, GC, GC);
  if(format == "COO" | format == "coo") {
    v.init_single(NC);
  } else {
    v.init_split(NC);
  }
  v.wait();

  std::uint64_t t_op_start = hpx::util::high_resolution_clock::now();
  tbsla::hpx_::Vector r;
  if(op == "spmv") {
    r = m->spmv(v);
  } else if(op == "a_axpx") {
    r = m->a_axpx_(v);
  }
  r.wait();
  std::uint64_t t_op_end = hpx::util::high_resolution_clock::now();
  std::uint64_t t_app_end = hpx::util::high_resolution_clock::now();

  std::map<std::string, std::string> outmap;
  outmap["test"] = op;
  outmap["matrix_format"] = format;
  outmap["n_row"] = std::to_string(m->get_n_row());
  outmap["n_col"] = std::to_string(m->get_n_col());
  outmap["g_row"] = std::to_string(GR);
  outmap["g_col"] = std::to_string(GC);
  outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
  outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
  outmap["lang"] = "HPX";
  if(matrix == "cdiag") {
    outmap["matrix_type"] = "cdiag";
    outmap["cdiag_c"] = std::to_string(C);
  } else if(matrix == "cqmat") {
    outmap["matrix_type"] = "cqmat";
    outmap["cqmat_c"] = std::to_string(C);
    outmap["cqmat_q"] = std::to_string(Q);
    outmap["cqmat_s"] = std::to_string(S);
  }

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
  desc_commandline.add_options()
    ("GR", value<std::uint64_t>()->default_value(10),
    "Number of submatrices in rows")
    ("GC", value<std::uint64_t>()->default_value(10),
    "Number of submatrices in columns")
    ("NR", value<std::uint64_t>()->default_value(1024),
    "Number of rows")
    ("NC", value<std::uint64_t>()->default_value(1024),
    "Number of columns")
    ("C", value<std::uint64_t>()->default_value(8),
    "Number of diagonals")
    ("Q", value<double>()->default_value(0.1),
    "Probability of column perturbation with cqmat")
    ("S", value<std::uint64_t>()->default_value(0),
    "Seed to generate cqmat")
    ("op", value<std::string>()->default_value(""),
    "operation performed (spmv or a_axpx_ (default : empty)")
    ("format", value<std::string>()->default_value(""),
    "storage format of the matrix (default : empty)")
    ("cqmat", "generate a cqmat matrix")
    ("cdiag", "generate a cdiag matrix");

  // Initialize and run HPX
  return hpx::init(desc_commandline, argc, argv);
}
