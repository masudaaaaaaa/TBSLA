#include <tbsla/cpp/Vector.h>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/mpi/vector_comm.h>
#include <vector>
#include <iostream>

C_CPP_Vector_t *C_MPI_allgatherv(MPI_Comm comm, C_CPP_Vector_t *v, int bn_row, int lgr) {
  std::cout << "bnrow: " << bn_row << "; lgr: " << lgr << std::endl;
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (v == NULL)
    return r;
  std::vector<double> *v_obj;
  v_obj = static_cast<std::vector<double> *>(v->obj);

  std::vector<int> recvcounts(lgr);
  std::vector<int> displs(lgr, 0);
  for(int i = 0; i < lgr; i++) {
    recvcounts[i] = tbsla::utils::range::lnv(bn_row, i, lgr);
  }
  for(int i = 1; i < lgr; i++) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }
  std::vector<double> recv(bn_row);
  MPI_Allgatherv(v_obj->data(), v_obj->size(), MPI_DOUBLE, recv.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);

  C_CPP_Vector_copy(r, &recv);
  return r;
}


C_CPP_Vector_t *C_MPI_reduce_sum(MPI_Comm comm, C_CPP_Vector_t *v, int n) {
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (v == NULL)
    return r;
  std::vector<double> *v_obj;
  v_obj = static_cast<std::vector<double> *>(v->obj);
  std::vector<double> r_obj(n);
  MPI_Allreduce(v_obj->data(), r_obj.data(), v_obj->size(), MPI_DOUBLE, MPI_SUM, comm);
  C_CPP_Vector_copy(r, &r_obj);
  return r;
}


C_CPP_Vector_t *C_MPI_reduce_gather(MPI_Comm comm, C_CPP_Vector_t *v, int bn_row, int bn_col, int lpr, int lpc, int lgr, int lgc) {
  C_CPP_Vector_t *r = C_CPP_Vector_create();
  if (v == NULL)
    return r;
  std::vector<double> *v_obj;
  v_obj = static_cast<std::vector<double> *>(v->obj);

  MPI_Comm row_comm;
  MPI_Comm_split(comm, lpr, lpc, &row_comm);
  std::vector<double> recv(v_obj->size());
  MPI_Allreduce(v_obj->data(), recv.data(), v_obj->size(), MPI_DOUBLE, MPI_SUM, row_comm);

  std::vector<int> recvcounts(lgr);
  std::vector<int> displs(lgr, 0);
  for(int i = 0; i < lgr; i++) {
    recvcounts[i] = tbsla::utils::range::lnv(bn_row, i, lgr);
  }
  for(int i = 1; i < lgr; i++) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }
  std::vector<double> recv2(bn_row);
  MPI_Comm col_comm;
  MPI_Comm_split(comm, lpc, lpr, &col_comm);
  MPI_Allgatherv(recv.data(), recv.size(), MPI_DOUBLE, recv2.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, col_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&row_comm);


  C_CPP_Vector_copy(r, &recv2);
  return r;
}

C_CPP_Vector_t *C_MPI_redistribute(MPI_Comm comm, C_CPP_Vector_t *v, int bn_row, int bn_col, int lpr, int lpc, int lgr, int lgc) {
  if(lgc == 1 && lgr == 1) {
    return v;
  } else if(lgc == 1 && lgr > 1) {
    return C_MPI_allgatherv(comm, v, bn_row, lgr);
  } else if(lgc > 1 && lgr == 1) {
    return C_MPI_reduce_sum(comm, v, bn_row);
  } else {
    return C_MPI_reduce_gather(comm, v, bn_row, bn_col, lpr, lpc, lgr, lgc);
  }
}

