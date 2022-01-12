#include <tbsla/cpp/MatrixSCOO.h>
#include <tbsla/cpp/MatrixSCOO.hpp>

struct C_CPP_MatrixSCOO {
  void *obj;
};


C_CPP_MatrixSCOO_t *C_CPP_MatrixSCOO_create() {
  C_CPP_MatrixSCOO_t *m;
  tbsla::cpp::MatrixSCOO *obj;

  m = (typeof(m))malloc(sizeof(*m));;
  obj = new tbsla::cpp::MatrixSCOO();
  m->obj = obj;

  return m;
}


void C_CPP_MatrixSCOO_destroy(C_CPP_MatrixSCOO_t *m) {
  if (m == NULL)
    return;
  delete static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  free(m);
}



void C_CPP_MatrixSCOO_fill_cdiag(C_CPP_MatrixSCOO_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  tbsla::cpp::MatrixSCOO *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  obj->fill_cdiag(n_row, n_col, cdiag, pr, pc, NR, NC);
}

void C_CPP_MatrixSCOO_fill_cqmat(C_CPP_MatrixSCOO_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  tbsla::cpp::MatrixSCOO *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  obj->fill_cqmat(n_row, n_col, c, q, seed_mult, pr, pc, NR, NC);
}

bool C_CPP_MatrixSCOO_read(C_CPP_MatrixSCOO_t *m, char *filename, int seek) {
  tbsla::cpp::MatrixSCOO *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  std::ifstream is(std::string(filename), std::ifstream::binary);
  if (!is.is_open())
    return false;
  is.seekg(seek, is.beg);
  obj->read(is);
  is.close();
  return true;
}

bool C_CPP_MatrixSCOO_write(C_CPP_MatrixSCOO_t *m, char *filename) {
  tbsla::cpp::MatrixSCOO *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  std::ofstream os(std::string(filename), std::ofstream::binary | std::ofstream::app);
  if (!os.is_open())
    return false;
  obj->write(os);
  os.close();
  return true;
}

void C_CPP_MatrixSCOO_print(C_CPP_MatrixSCOO_t *m) {
  tbsla::cpp::MatrixSCOO *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  obj->print_infos(std::cerr);
  std::cerr << *obj << std::endl;
}

C_CPP_Vector_t *C_CPP_MatrixSCOO_spmv(C_CPP_MatrixSCOO_t *m, C_CPP_Vector_t *v) {
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (m == NULL || v == NULL)
    return r;
  tbsla::cpp::MatrixSCOO *m_obj;
  std::vector<double> *v_obj;
  m_obj = static_cast<tbsla::cpp::MatrixSCOO *>(m->obj);
  v_obj = static_cast<std::vector<double> *>(v->obj);
  std::vector<double> r_obj = m_obj->spmv(*v_obj, 0);
  C_CPP_Vector_copy(r, &r_obj);
  return r;
}
