#include <tbsla/cpp/MatrixDENSE.h>
#include <tbsla/cpp/MatrixDENSE.hpp>

struct C_CPP_MatrixDENSE {
  void *obj;
};


C_CPP_MatrixDENSE_t *C_CPP_MatrixDENSE_create() {
  C_CPP_MatrixDENSE_t *m;
  tbsla::cpp::MatrixDENSE *obj;

  m = (typeof(m))malloc(sizeof(*m));;
  obj = new tbsla::cpp::MatrixDENSE();
  m->obj = obj;

  return m;
}


void C_CPP_MatrixDENSE_destroy(C_CPP_MatrixDENSE_t *m) {
  if (m == NULL)
    return;
  delete static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  free(m);
}



void C_CPP_MatrixDENSE_fill_cdiag(C_CPP_MatrixDENSE_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  tbsla::cpp::MatrixDENSE *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  obj->fill_cdiag(n_row, n_col, cdiag, pr, pc, NR, NC);
}

void C_CPP_MatrixDENSE_fill_cqmat(C_CPP_MatrixDENSE_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  tbsla::cpp::MatrixDENSE *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  obj->fill_cqmat(n_row, n_col, c, q, seed_mult, pr, pc, NR, NC);
}

bool C_CPP_MatrixDENSE_read(C_CPP_MatrixDENSE_t *m, char *filename, int seek) {
  tbsla::cpp::MatrixDENSE *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  std::ifstream is(std::string(filename), std::ifstream::binary);
  if (!is.is_open())
    return false;
  is.seekg(seek, is.beg);
  obj->read(is);
  is.close();
  return true;
}

bool C_CPP_MatrixDENSE_write(C_CPP_MatrixDENSE_t *m, char *filename) {
  tbsla::cpp::MatrixDENSE *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  std::ofstream os(std::string(filename), std::ofstream::binary | std::ofstream::app);
  if (!os.is_open())
    return false;
  obj->write(os);
  os.close();
  return true;
}

void C_CPP_MatrixDENSE_print(C_CPP_MatrixDENSE_t *m) {
  tbsla::cpp::MatrixDENSE *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  obj->print_infos(std::cerr);
  std::cerr << *obj << std::endl;
}

C_CPP_Vector_t *C_CPP_MatrixDENSE_spmv(C_CPP_MatrixDENSE_t *m, C_CPP_Vector_t *v) {
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (m == NULL || v == NULL)
    return r;
  tbsla::cpp::MatrixDENSE *m_obj;
  std::vector<double> *v_obj;
  m_obj = static_cast<tbsla::cpp::MatrixDENSE *>(m->obj);
  v_obj = static_cast<std::vector<double> *>(v->obj);
  std::vector<double> r_obj = m_obj->spmv(*v_obj, 0);
  C_CPP_Vector_copy(r, &r_obj);
  return r;
}
