#include <tbsla/cpp/MatrixELL.h>
#include <tbsla/cpp/MatrixELL.hpp>

struct C_CPP_MatrixELL {
  void *obj;
};


C_CPP_MatrixELL_t *C_CPP_MatrixELL_create() {
  C_CPP_MatrixELL_t *m;
  tbsla::cpp::MatrixELL *obj;

  m = (typeof(m))malloc(sizeof(*m));;
  obj = new tbsla::cpp::MatrixELL();
  m->obj = obj;

  return m;
}


void C_CPP_MatrixELL_destroy(C_CPP_MatrixELL_t *m) {
  if (m == NULL)
    return;
  delete static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  free(m);
}



void C_CPP_MatrixELL_fill_cdiag(C_CPP_MatrixELL_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  tbsla::cpp::MatrixELL *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  obj->fill_cdiag(n_row, n_col, cdiag, pr, pc, NR, NC);
}

void C_CPP_MatrixELL_fill_cqmat(C_CPP_MatrixELL_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  tbsla::cpp::MatrixELL *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  obj->fill_cqmat(n_row, n_col, c, q, seed_mult, pr, pc, NR, NC);
}

bool C_CPP_MatrixELL_read(C_CPP_MatrixELL_t *m, char *filename, int seek) {
  tbsla::cpp::MatrixELL *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  std::ifstream is(std::string(filename), std::ifstream::binary);
  if (!is.is_open())
    return false;
  is.seekg(seek, is.beg);
  obj->read(is);
  is.close();
  return true;
}

bool C_CPP_MatrixELL_write(C_CPP_MatrixELL_t *m, char *filename) {
  tbsla::cpp::MatrixELL *obj;
  if (m == NULL)
    return false;
  obj = static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  std::ofstream os(std::string(filename), std::ofstream::binary | std::ofstream::app);
  if (!os.is_open())
    return false;
  obj->write(os);
  os.close();
  return true;
}

void C_CPP_MatrixELL_print(C_CPP_MatrixELL_t *m) {
  tbsla::cpp::MatrixELL *obj;
  if (m == NULL)
    return;
  obj = static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  obj->print_infos(std::cerr);
  std::cerr << *obj << std::endl;
}

C_CPP_Vector_t *C_CPP_MatrixELL_spmv(C_CPP_MatrixELL_t *m, C_CPP_Vector_t *v) {
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (m == NULL || v == NULL)
    return r;
  tbsla::cpp::MatrixELL *m_obj;
  std::vector<double> *v_obj;
  m_obj = static_cast<tbsla::cpp::MatrixELL *>(m->obj);
  v_obj = static_cast<std::vector<double> *>(v->obj);
  std::vector<double> r_obj = m_obj->spmv(*v_obj, 0);
  C_CPP_Vector_copy(r, &r_obj);
  return r;
}
