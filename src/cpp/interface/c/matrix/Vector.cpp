#include <tbsla/cpp/Vector.h>
#include <tbsla/cpp/utils/vector.hpp>
#include <stdlib.h>

#include <vector>
#include <fstream>
#include <string>
#include <numeric>
#include <algorithm>



C_CPP_Vector_t *C_CPP_Vector_create() {
  C_CPP_Vector_t *v = NULL;
  std::vector<double> *obj = NULL;

  v = (typeof(v))malloc(sizeof(*v));
  obj = new std::vector<double>();
  v->obj = obj;

  return v;
}

void C_CPP_Vector_destroy(C_CPP_Vector_t *v) {
  if (v == NULL)
    return;
  delete static_cast<std::vector<double> *>(v->obj);
  free(v);
}

void C_CPP_Vector_print(C_CPP_Vector_t *v) {
  std::vector<double> *obj = NULL;
  if (v == NULL) {
    printf("C_CPP_Vector_print : input vector is NULL\n");
    return;
  }
  obj = static_cast<std::vector<double> *>(v->obj);
  tbsla::utils::vector::streamvector<double>(std::cout, "v", *obj);
  std::cout << std::endl << std::flush;
}

void C_CPP_Vector_set(C_CPP_Vector_t *v, void *obj) {
  if (v == NULL)
    return;
/*
  if (v->obj != NULL)
    delete static_cast<std::vector<double> *>(v->obj);
*/
  v->obj = obj;
}

void C_CPP_Vector_copy(C_CPP_Vector_t *v, void *obj) {
  std::vector<double> *v_obj = NULL;
  std::vector<double> *i_obj = NULL;
  if (v == NULL)
    return;
  v_obj = static_cast<std::vector<double> *>(v->obj);
  i_obj = static_cast<std::vector<double> *>(obj);
  v_obj->resize(i_obj->size());
  v_obj->assign(i_obj->begin(), i_obj->end());
}

void C_CPP_Vector_fill(C_CPP_Vector_t *v, int n, int s) {
  std::vector<double> *obj = NULL;
  if (v == NULL)
    return;
  obj = static_cast<std::vector<double> *>(v->obj);
  obj->clear();
  obj->resize(n);
  std::iota (std::begin(*obj), std::end(*obj), s);
}

bool C_CPP_Vector_read(C_CPP_Vector_t *v, char *filename, int seek) {
  std::vector<double> *obj = NULL;
  if (v == NULL)
    return false;
  obj = static_cast<std::vector<double> *>(v->obj);
  std::ifstream is(std::string(filename), std::ifstream::binary);
  if (!is.is_open())
    return false;
  is.seekg(seek, is.beg);
  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  obj->resize(size);
  is.read(reinterpret_cast<char*>(obj->data()), size * sizeof(double));
  is.close();
  return true;
}

bool C_CPP_Vector_write(C_CPP_Vector_t *v, char *filename) {
  std::vector<double> *obj = NULL;
  if (v == NULL)
    return false;
  obj = static_cast<std::vector<double> *>(v->obj);
  std::ofstream os(std::string(filename), std::ofstream::binary | std::ofstream::app);
  if (!os.is_open())
    return false;
  size_t size = obj->size();
  os.write(reinterpret_cast<char*>(&size), sizeof(size));
  os.write(reinterpret_cast<char*>(obj->data()), size * sizeof(double));
  os.close();
  return true;
}

bool C_CPP_Vector_add(C_CPP_Vector_t *v1, C_CPP_Vector_t *v2) {
  std::vector<double> *v1obj = NULL;
  std::vector<double> *v2obj = NULL;
  if (v1 == NULL || v2 == NULL)
    return false;
  v1obj = static_cast<std::vector<double> *>(v1->obj);
  v2obj = static_cast<std::vector<double> *>(v2->obj);
  std::transform(v1obj->begin(), v1obj->end(), v2obj->begin(), v1obj->begin(), std::plus<double>());
  return true;
}

bool C_CPP_Vector_add_incr(C_CPP_Vector_t *v1, C_CPP_Vector_t *v2, int incr) {
  std::vector<double> *v1obj = NULL;
  std::vector<double> *v2obj = NULL;
  if (v1 == NULL || v2 == NULL)
    return false;
  v1obj = static_cast<std::vector<double> *>(v1->obj);
  v2obj = static_cast<std::vector<double> *>(v2->obj);
  std::transform(v1obj->begin(), v1obj->end(), v2obj->begin() + incr, v1obj->begin(), std::plus<double>());
  return true;
}

bool C_CPP_Vector_gather(C_CPP_Vector_t *v1, C_CPP_Vector_t *v2) {
  std::vector<double> *v1obj = NULL;
  std::vector<double> *v2obj = NULL;
  if (v1 == NULL || v2 == NULL)
    return false;
  v1obj = static_cast<std::vector<double> *>(v1->obj);
  v2obj = static_cast<std::vector<double> *>(v2->obj);
  v1obj->insert(v1obj->end(), v2obj->begin(), v2obj->end());
  return true;
}
