#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <mpi.h>
#include <iostream>

long int const tbsla::mpi::Matrix::compute_sum_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  return nnz;
}

long int const tbsla::mpi::Matrix::compute_min_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  return nnz;
}

long int const tbsla::mpi::Matrix::compute_max_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  return nnz;
}

double* tbsla::mpi::Matrix::spmv_no_redist(MPI_Comm comm, const double* v, int vect_incr) {
  return this->spmv(v, vect_incr);
}

inline void tbsla::mpi::Matrix::Ax_(MPI_Comm comm, double* r, const double* v, int vect_incr) {
  this->Ax(r, v, vect_incr);
}

/*
 * comm : MPI communicator
 * r : results (size : n_row)
 * v : input vector (size : n_col)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row)
 *
 */
inline void tbsla::mpi::Matrix::Ax(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, int vect_incr) {
  this->Ax(buffer, v, vect_incr);
  if(this->NC == 1 && this->NR > 1) {
    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(buffer, this->ln_row, MPI_DOUBLE, r, recvcounts, displs, MPI_DOUBLE, comm);
  } else if(this->NC > 1 && this->NR == 1) {
    MPI_Allreduce(buffer, r, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(buffer, buffer2, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(buffer2, this->ln_row, MPI_DOUBLE, r, recvcounts, displs, MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
  }
}


/*
 * comm : MPI communicator
 * r : results (size : ln_col)
 * v : input vector (size : ln_col)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row)
 *
 */
inline void tbsla::mpi::Matrix::Ax_local(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, int vect_incr) {
  this->Ax(buffer, v, vect_incr);
  if(this->NC == 1) {
    for(int k=0; k<this->ln_row; k++)
      buffer2[k] = buffer[k];
  } else if(this->NC > 1 && this->NR == 1) {
    MPI_Allreduce(buffer, buffer2, this->ln_row, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(buffer, buffer2, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);
    MPI_Comm_free(&row_comm);
  }
}


inline double tbsla::mpi::Matrix::pagerank_normalization(MPI_Comm comm, double* b, double* b_t, double beta) {
  double sum = 0;
  #pragma omp parallel for reduction(+ : sum) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    sum += b_t[k];
  }
  double t_sum;
  if(this->NC == 1) {
    t_sum = sum;
  }
  else {
    // reduce to have sum of whole vector
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(&sum, &t_sum, 1, MPI_DOUBLE, MPI_SUM, row_comm);
    MPI_Comm_free(&row_comm);
  }
  t_sum *= (1-beta)/this->n_col;
  double max_val = 0;
  #pragma omp parallel for reduction(max : max_val) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    b[k] = beta*b[k] + t_sum;
    if(b[k] > max_val)
      max_val = b[k];
  }
  double max_overall;
  MPI_Allreduce(&max_val, &max_overall, 1, MPI_DOUBLE, MPI_MAX, comm);
  double error = 0;
  #pragma omp parallel for reduction(+ : error) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    b[k] /= max_overall;
    error += std::abs(b[k]- b_t[k]);
  }
  double error_overall;
  MPI_Allreduce(&error, &error_overall, 1, MPI_DOUBLE, MPI_SUM, comm);
  //quick fix since partial vectors are each duplicated on NC block processes...
  return (error_overall/this->NC);
}


void tbsla::mpi::Matrix::pagerank_norma_end(MPI_Comm comm, double* b) {
  double sum = 0;
  #pragma omp parallel for reduction(+ : sum) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    sum += b[k];
  }
  double t_sum;
  //MPI_Allreduce(&sum, &t_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  if(this->NC == 1) {
    t_sum = sum;
  }
  else {
    // reduce to have sum of whole vector
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allreduce(&sum, &t_sum, 1, MPI_DOUBLE, MPI_SUM, col_comm);
    MPI_Comm_free(&col_comm);
  }
  #pragma omp parallel for reduction(+ : sum) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    b[k] /= t_sum;
  }
}


std::unordered_map<int,std::vector<int> > tbsla::mpi::Matrix::find_senders_receivers(MPI_Comm comm) {
  std::unordered_map<int,std::vector<int> > res_map;
  int recv_start = this->f_col;
  int recv_end = this->f_col + this->ln_col;
  for(int k=0; k<this->NR; k++) {
    int rowblock_start = k * this->ln_row;
    int rowblock_end = rowblock_start + this->ln_row;
    if(rowblock_start <= recv_end && rowblock_end >= recv_start
        && (this->pc != 0 || this->pr != k)) {
      int s_start, r_start, length;
      if(rowblock_start <= recv_start) {
        r_start = 0;
        s_start = recv_start - rowblock_start;
      }
      else {
        r_start = rowblock_start - recv_start;
        s_start = 0;
      }
      int endpoint_abs = std::min(rowblock_end, recv_end);
      int startpoint_abs = recv_start + r_start;
      length = endpoint_abs - startpoint_abs;
      if(length > 0) {
        std::vector<int> coords;
        coords.push_back(s_start); coords.push_back(r_start); coords.push_back(length);
        res_map[k] = coords;
        //std::cout << "need to receive from process at pr=" << k << " : " << s_start << " | " << r_start << " | " << length << std::endl;
      }
    }
  }
  return res_map;
}


std::vector<MPI_Comm> tbsla::mpi::Matrix::create_comms(MPI_Comm comm, std::unordered_map<int,std::vector<int> > recv_map) {
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  std::vector<MPI_Comm> res_vec;
  // if process is sender/receiver => color = 1, else color = MPI_UNDEFINED
  // key (rank) = 0 if sender ; <iterator> (pc) if receiver
  for(int k=0; k<this->NR; k++) {
    MPI_Comm new_comm;
    int color = MPI_UNDEFINED, key = (this->pr) + 1;
    if(this->pc == 0 && this->pr == k) {
      color = 0;
      key = 0;
    }
    else if(recv_map.find(k) != recv_map.end()) {
      color = 0;
    }
    std::cout << "k = " << k << " => process at " << this->pr << "|" << this->pc << " => " << color << "|" << key << std::endl;
    MPI_Comm_split(comm, color, key, &new_comm);
    res_vec.push_back(new_comm);
  }
  return res_vec;
}





/*
 * comm : MPI communicator
 * r : results (size : ln_col)
 * v : input vector (size : ln_col)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row) : contains local Ax results (reduced on cols)
 *
 */
void tbsla::mpi::Matrix::redistribute_vector(std::vector<MPI_Comm> comms, double* r, const double* v, double* buffer, double* buffer2, std::unordered_map<int,std::vector<int> > recv_map) {
  for(int k=0; k<this->NR; k++) {
    //std::cout << "k = " << k << std::endl;
    // can reuse 'buffer' as recv => no need
    // change parts of 'r' as needed
    bool is_sender = (this->pr == k && this->pc == 0);
    bool is_recv = (recv_map.find(k) != recv_map.end()) && !is_sender;
    //std::cout << "is_sender = " << is_sender << " | is_recv = " << is_recv << std::endl;
    MPI_Barrier(comms[k]);
    if(is_sender || is_recv) {
      MPI_Bcast(buffer2, this->ln_row, MPI_DOUBLE, 0, comms[k]);
      if(is_recv) {
        // 's_start', 'r_start', 'length'
        std::vector<int> bounds = recv_map[k];
        //std::cout << "process at " << this->pr << "|" << this->pc << " receiving " << bounds[2] << " elements ; s_start = " << bounds[0] << " | r_start = " << bounds[1] << std::endl;
        for(int z=0; z<bounds[2]; z++)
          r[bounds[1]+z] = buffer2[bounds[0]+z];
      }
      else if(is_sender) {
        //std::cout << "process at " << this->pr << "|" << this->pc << " sending " << this->ln_row << " elements" << std::endl;
      }
    }
  }
  // also copy if there is overlap between output and input (of next iteration)
  int recv_start = this->f_col;
  int recv_end = this->f_col + this->ln_col;
  int rowblock_start = this->f_row;
  int rowblock_end = this->f_row + this->ln_row;
  if(recv_start <= rowblock_end && recv_end >= rowblock_start) {
    /*int copy_start = std::max(inp_start, outp_start);
    int copy_end = std::min(inp_end, outp_end);
    int copy_start = std::max(inp_start, outp_start);
    for(int z=copy_start; z<copy_end; z++)
      r[z] = buffer2[z];*/
    int s_start, r_start, length;
    if(rowblock_start <= recv_start) {
      r_start = 0;
      s_start = recv_start - rowblock_start;
    }
    else {
      r_start = rowblock_start - recv_start;
      s_start = 0;
    }
    int endpoint_abs = std::min(rowblock_end, recv_end);
    int startpoint_abs = recv_start + r_start;
    length = endpoint_abs - startpoint_abs;
    //std::cout << "send_start = " << s_start << " ; rec_start = " << r_start << " ; length = " << length << std::endl;
    for(int z=0; z<length; z++) {
      r[r_start+z] = buffer2[s_start+z];
    }
  }
}

/*
 * comm : MPI communicator
 * s : sum of rows ; internal use (size : n_row)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row)
 *
 */
/*inline void tbsla::mpi::Matrix::make_stochastic(MPI_Comm comm, double* s, double* buffer, double* buffer2) {
  this->get_row_sums(buffer);
  if(this->NC == 1 && this->NR > 1) {
	int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(buffer, this->ln_row, MPI_DOUBLE, s, recvcounts, displs, MPI_DOUBLE, comm);
  } else if(this->NC > 1 && this->NR == 1) {
    MPI_Allreduce(buffer, s, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(buffer, buffer2, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(buffer2, this->ln_row, MPI_DOUBLE, s, recvcounts, displs, MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
  }
  this->normalize_rows(s);
}*/

/*
 * comm : MPI communicator
 * s : sum of cols ; internal use (size : n_col)
 * buffer : buffer for internal operations (size : ln_col)
 * buffer2 : buffer for internal operations (size : ln_col)
 *
 */
// normalize on columns instead

inline void tbsla::mpi::Matrix::make_stochastic(MPI_Comm comm, double* s, double* buffer, double* buffer2) {
  this->get_col_sums(buffer);
  std::cout << "computed col_sums" << std::endl;
  if(this->NR == 1) {
    for(int k=0; k<this->ln_col; k++)
      s[k] = buffer[k];
  }
  else if(this->NC == 1 && this->NR > 1) {
    MPI_Allreduce(buffer, s, this->n_col, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    std::cout << "NR > 1 and NC > 1" << std::endl;
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allreduce(buffer, s, this->ln_col, MPI_DOUBLE, MPI_SUM, col_comm);

    MPI_Comm_free(&col_comm);
    std::cout << "end" << std::endl;
  }
  double tot = 0;
  for(int i=0; i<this->ln_col; i++) {
    tot += s[i];
  }
  std::cout << "tot = " << tot << std::endl;
  this->normalize_cols(s);
}

double* tbsla::mpi::Matrix::spmv(MPI_Comm comm, const double* v, int vect_incr) {
  double* send = this->spmv(v, vect_incr);
  if(this->NC == 1 && this->NR == 1) {
    return send;
  } else if(this->NC == 1 && this->NR > 1) {
    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    double* recv = new double[this->n_row]();
    MPI_Allgatherv(send, this->ln_row, MPI_DOUBLE, recv, recvcounts, displs, MPI_DOUBLE, comm);
    return recv;
  } else if(this->NC > 1 && this->NR == 1) {
    double* recv = new double[this->n_row]();
    MPI_Allreduce(send, recv, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
    return recv;
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    double* recv = new double[this->ln_row]();
    MPI_Allreduce(send, recv, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    double* recv2 = new double[this->n_row]();
    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(recv, this->ln_row, MPI_DOUBLE, recv2, recvcounts, displs, MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    return recv2;
  }
}

double* tbsla::mpi::Matrix::a_axpx_(MPI_Comm comm, const double* v, int vect_incr) {
  double* r = this->spmv(comm, v + this->f_col, vect_incr);
  std::transform (r, r + this->n_row, v, r, std::plus<double>());
  double* r2 = this->spmv(comm, r + this->f_col, vect_incr);
  delete[] r;
  return r2;
}

inline void tbsla::mpi::Matrix::AAxpAx(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, double* buffer3, int vect_incr) {
  this->Ax(comm, buffer3, v + this->f_col, buffer, buffer2, vect_incr);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < this->n_row; i++) {
    buffer3[i] += v[i];
  }
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < this->ln_row; i++) {
    buffer[i] = 0;
    buffer2[i] = 0;
  }
  this->Ax(comm, r, buffer3 + this->f_col, buffer, buffer2, vect_incr);
}

inline void tbsla::mpi::Matrix::AAxpAxpx(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, double* buffer3, int vect_incr) {
  this->AAxpAx(comm, r, v, buffer, buffer2, buffer3, vect_incr);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < this->n_row; i++) {
    r[i] += v[i];
  }
}

/*double* tbsla::mpi::Matrix::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done){
  //std::cout << "PageRank" << std::endl;
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  double* b = new double[n_col];
  double* buf1 = new double[ln_row];
  double* buf2 = new double[ln_row];
  double* b_t = new double[n_col];
  //std::cout << "outer-one" << std::endl;
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < n_col; i++){
    b[i] = 1;
  }
  bool converge = false;
  int nb_iterations = 0;
  double max_val, error, teleportation_sum;

  while(!converge && nb_iterations < max_iterations){
    //std::cout << "inner-one" << std::endl;
    //std::cout << nb_iterations << std::endl;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < n_col; i++) {
      b_t[i] = b[i];
      b[i] = 0;
    }
    //std::cout << "inner-two" << std::endl;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < ln_row; i++) {
      buf1[i] = 0;
      buf2[i] = 0;
    }
    //std::cout << "inner-three" << std::endl;
    this->Ax(comm, b, b_t + f_col, buf1, buf2);
    //std::cout << "inner-four" << std::endl;
    #pragma omp parallel for reduction(+ : teleportation_sum) schedule(static)
    for(int i = 0 ; i < n_col; i++){
      teleportation_sum += b_t[i];
    }
    teleportation_sum *= (1-beta)/n_col ;

    b[0] = beta*b[0] + teleportation_sum;
    max_val = b[0];
	error = 0.0;
    //std::cout << "inner-five" << std::endl;
    #pragma omp parallel for reduction(max : max_val) schedule(static)
    for(int  i = 1 ; i < n_col;i++){
      b[i] = beta*b[i] + teleportation_sum;
      if(max_val < b[i])
        max_val = b[i];
      //error += std::abs(b[i]- b_t[i]);
    }

	// no need to normalize here
    error = 0.0;
    //std::cout << "inner-six" << std::endl;
    #pragma omp parallel for reduction(+ : error) schedule(static)
    for (int i = 0;i< n_col;i++){
      b[i] = b[i]/max_val;
      error += std::abs(b[i]- b_t[i]);
    }
    if(error < epsilon){
      converge = true;
    }
    nb_iterations++;
  }

  nb_iterations_done = nb_iterations;

  //std::cout << "outer-two" << std::endl;
  double sum = b[0];
  #pragma omp parallel for reduction(+ : sum) schedule(static)
  for(int i = 1; i < n_col; i++) {
    sum += b[i];
  }

  //std::cout << "outer-three" << std::endl;
  #pragma omp parallel for schedule(static)
  for(int i = 0 ; i < n_col; i++) {
    b[i] = b[i]/sum;
  }
  delete[] b_t;
  delete[] buf1;
  delete[] buf2;
  return b;
}*/


// Alternative version removing the need to store the full result vector on each process
// TODO : fix/debug
double* tbsla::mpi::Matrix::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done){
  //std::cout << "PageRank" << std::endl;
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  double* b = new double[ln_col];
  double* buf1 = new double[ln_row];
  double* buf2 = new double[ln_row];
  double* b_t = new double[ln_col];
  //std::cout << "outer-one" << std::endl;
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < ln_col; i++){
    b[i] = 1;
  }

  std::unordered_map<int,std::vector<int> > recv_map = this->find_senders_receivers(comm);
  std::vector<MPI_Comm> comms = this->create_comms(comm, recv_map);

  bool converge = false;
  int nb_iterations = 0;
  double max_val, error, teleportation_sum;

  while(!converge && nb_iterations < max_iterations){
    //std::cout << "inner-one" << std::endl;
    //std::cout << nb_iterations << std::endl;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < ln_col; i++) {
      b_t[i] = b[i];
      b[i] = 0;
    }
    //std::cout << "inner-two" << std::endl;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < ln_row; i++) {
      buf1[i] = 0;
      buf2[i] = 0;
    }
    //std::cout << "inner-three" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    //this->Ax_local(comm, b, b_t + f_col, buf1, buf2);
    this->Ax_local(comm, b, b_t, buf1, buf2);
    //std::cout << "inner-four" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    this->redistribute_vector(comms, b, b_t, buf1, buf2, recv_map);
    //std::cout << "inner-five" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    error = pagerank_normalization(comm, b, b_t, beta);
    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout << "error = " << error << std::endl;
    if(error < epsilon){
      converge = true;
    }
    nb_iterations++;
  }


  nb_iterations_done = nb_iterations;

  //std::cout << "outer-two" << std::endl;
  this->pagerank_norma_end(comm, b);
  //std::cout << "outer-three" << std::endl;
  delete[] b_t;
  delete[] buf1;
  return b;
}


