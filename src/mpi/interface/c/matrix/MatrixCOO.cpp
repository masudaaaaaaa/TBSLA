#include <tbsla/mpi/MatrixCOO.h>
#include <tbsla/mpi/MatrixCOO.hpp>

struct C_MPI_MatrixCOO {
  void *obj;
};


C_MPI_MatrixCOO_t *C_MPI_MatrixCOO_create() {
  C_MPI_MatrixCOO_t *m;
  tbsla::mpi::MatrixCOO *obj;

  m = (typeof(m))malloc(sizeof(*m));;
  obj = new tbsla::mpi::MatrixCOO();
  m->obj = obj;

  return m;
}


void C_MPI_MatrixCOO_destroy(C_MPI_MatrixCOO_t *m) {
  if (m == NULL)
    return;
  delete static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  free(m);
}



void C_MPI_MatrixCOO_fill_cdiag(C_MPI_MatrixCOO_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  tbsla::mpi::MatrixCOO *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  obj->fill_cdiag(n_row, n_col, cdiag, pr, pc, NR, NC);
}

void C_MPI_MatrixCOO_fill_cqmat(C_MPI_MatrixCOO_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  tbsla::mpi::MatrixCOO *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  obj->fill_cqmat(n_row, n_col, c, q, seed_mult, pr, pc, NR, NC);
}

bool C_MPI_MatrixCOO_read(C_MPI_MatrixCOO_t *m, char *filename, int seek) {
  tbsla::mpi::MatrixCOO *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  std::ifstream is(std::string(filename), std::ifstream::binary);
  if (!is.is_open())
    return false;
  is.seekg(seek, is.beg);
  obj->read(is);
  is.close();
  return true;
}

bool C_MPI_MatrixCOO_write(C_MPI_MatrixCOO_t *m, char *filename) {
  tbsla::mpi::MatrixCOO *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  std::ofstream os(std::string(filename), std::ofstream::binary | std::ofstream::app);
  if (!os.is_open())
    return false;
  obj->write(os);
  os.close();
  return true;
}

void C_MPI_MatrixCOO_print(C_MPI_MatrixCOO_t *m) {
  tbsla::mpi::MatrixCOO *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  obj->print_infos(std::cerr);
  std::cerr << *obj << std::endl;
}

C_CPP_Vector_t *C_MPI_MatrixCOO_spmv(C_MPI_MatrixCOO_t *m, MPI_Comm comm, C_CPP_Vector_t *v) {
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (m == NULL || v == NULL)
    return r;
  tbsla::mpi::MatrixCOO *m_obj;
  std::vector<double> *v_obj;
  m_obj = static_cast<tbsla::mpi::MatrixCOO *>(m->obj);
  v_obj = static_cast<std::vector<double> *>(v->obj);
  std::vector<double> r_obj = m_obj->spmv(comm, *v_obj, 0);
  C_CPP_Vector_copy(r, &r_obj);
  return r;
}
