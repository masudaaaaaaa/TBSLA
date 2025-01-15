#ifndef TBSLA_CPP_MatrixELL
#define TBSLA_CPP_MatrixELL

#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <iostream>
#include <fstream>

namespace tbsla { namespace cpp {

class MatrixELL : public virtual Matrix {
  public:
    MatrixELL() : values(0), columns(0), max_col(0) {};
    MatrixELL(const tbsla::cpp::MatrixCOO & m);
    ~MatrixELL();
    friend std::ostream & operator<<( std::ostream &os, const MatrixELL &m);
    double* spmv(const double* v, int vect_incr = 0) const;
    inline void Ax(double* r, const double* v, int vect_incr = 0) const;
    using tbsla::cpp::Matrix::a_axpx_;
    using tbsla::cpp::Matrix::AAxpAx;
    using tbsla::cpp::Matrix::AAxpAxpx;
    std::string get_vectorization() const;
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    std::ostream& print(std::ostream& os) const;
    void NUMAinit();
    void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_brain(int n_row, int n_col, int* neuron_type, std::vector<std::vector<double> > proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > brain_struct, unsigned int seed_mult, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
	
	void get_row_sums(double* buffer);
	void normalize_rows(double* buffer);
	void get_col_sums(double* buffer);
	void normalize_cols(double* buffer);
	void get_row_max_abs(double* max_abs);
        void apply_exponential(double* max_abs, int base);

  protected:
    double* values;
    int* columns;
    int max_col;
};

}}

#endif
