#ifndef TBSLA_CINTERFACE_CPP_Vector
#define TBSLA_CINTERFACE_CPP_Vector

#ifdef __cplusplus
extern "C" {
#endif

struct C_CPP_Vector {
  void *obj;
};
typedef struct C_CPP_Vector C_CPP_Vector_t;

C_CPP_Vector_t *C_CPP_Vector_create();
void C_CPP_Vector_destroy(C_CPP_Vector_t *v);
void C_CPP_Vector_set(C_CPP_Vector_t *v, void *obj);
void C_CPP_Vector_copy(C_CPP_Vector_t *v, void *obj);

void C_CPP_Vector_print(C_CPP_Vector_t *v);
void C_CPP_Vector_fill(C_CPP_Vector_t *v, int n, int s);
bool C_CPP_Vector_read(C_CPP_Vector_t *v, char *filename, int seek);
bool C_CPP_Vector_write(C_CPP_Vector_t *v, char *filename);

bool C_CPP_Vector_add(C_CPP_Vector_t *v1, C_CPP_Vector_t *v2);
bool C_CPP_Vector_add_incr(C_CPP_Vector_t *v1, C_CPP_Vector_t *v2, int incr);
bool C_CPP_Vector_gather(C_CPP_Vector_t *v1, C_CPP_Vector_t *v2);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_CINTERFACE_CPP_Vector */
