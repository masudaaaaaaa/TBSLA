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
  double sum_val = 0;
  #pragma omp parallel for reduction(+ : sum_val) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    b[k] = beta*b[k] + t_sum;
    sum_val += b[k];
  }
  double sum_overall;
  MPI_Comm col_comm;
  MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
  MPI_Allreduce(&sum, &sum_overall, 1, MPI_DOUBLE, MPI_SUM, col_comm);
  MPI_Comm_free(&col_comm);
  //std::cout << "[sum_overall] somme vecteur résultat = " << sum_overall << std::endl;
  double error = 0;
  #pragma omp parallel for reduction(+ : error) schedule(static)
  for(int k=0; k<this->ln_col; k++) {
    b[k] /= sum_overall;
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


/*
 * comm : MPI communicator
 * s : sum of rows ; internal use (size : n_row)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row)
 *
 */
// set diagonalle at sum of row

inline void tbsla::mpi::Matrix::make_diagonally_dominant(MPI_Comm comm, double* s, double* buffer, double* buffer2) {
  this->get_row_sums(buffer);
  std::cout << "computed col_sums" << std::endl;
  if(this->NC == 1) {
    for(int k=0; k<this->ln_row; k++)
      s[k] = buffer[k];
  }
  else if(this->NR == 1 && this->NC > 1) {
    MPI_Allreduce(buffer, s, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    std::cout << "NR > 1 and NC > 1" << std::endl;
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(buffer, s, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    MPI_Comm_free(&row_comm);
    std::cout << "end" << std::endl;
  }
  double tot = 0;
  for(int i=0; i<this->ln_row; i++) {
    tot += s[i];
  }
  std::cout <<"s[1]= "<< s[1] << "tot = " << tot << std::endl;
  this->set_diag(s);
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


double abs_two_vector_error(double *vect1, double *vect2, int size)
{
    /*Calculate the error (norm) between two vectors of size "size"*/
    double sum=0;
    for (int i=0;i<size;i++)
    {sum += std::abs(vect1[i] - vect2[i]);}
    return sum;
}

double abs_one_vector_error(double *vect1,  int size)
{
    /*Calculate the error (norm) between two vectors of size "size"*/
    double sum=0;
    for (int i=0;i<size;i++)
    {sum += std::abs(vect1[i] - 0);}
    return sum;
}

/*
Variables used in PageRank and what they correspond to :
(See help PDF for more info)

    int indl; //Indice de ligne du block
    int indc; //Indice de colonne du block
    long dim_l; //nombre de lignes dans le block
    long dim_c; //nombre de colonnes dans le block
    long startRow; //Indice de départ en ligne (inclu)
    long startColumn; //Indice de départ en colonne (inclu)
    long endRow; //Indice de fin en ligne (inclu)
    long endColumn; //Indice de fin en colonne (inclu)

    int pr_result_redistribution_root; //Indice de colonne du block "root" (source) de la communication-redistribution du vecteur résultat
    int result_vector_calculation_group; //Indice de groupe de calcul du vecteur résultat
    long local_result_vector_size; //Taille locale du vecteur résultat du PageRank, en nombre d'éléments
    int indl_in_result_vector_calculation_group; //Indice de ligne du block dans le groupe de calcul du vecteur résultat
    int indc_in_result_vector_calculation_group; //Indice de colonne du block dans le groupe de calcul du vecteur résultat
    int inter_result_vector_need_group_communicaton_group; //Indice du Groupe de communication inter-groupe de besoin (utile pour récupérer le résultat final)
    long startColumn_in_result_vector_calculation_group; //Indice de départ en colonne dans le groupe de calcul du vecteur résultat (inclu)
    long startRow_in_result_vector_calculation_group; //Indice de départ en ligne dans le groupe de calcul du vecteur résultat (inclu), utile dans le PageRank pour aller chercher des valeurs dans le vecteur q
    int my_result_vector_calculation_group_rank; //my_rank dans le groupe de calcul du vecteur résultat
*/


// Alternative version optimizing the communications for Torus node allocation method (on Fugaku)
// TODO : fix/debug
double * tbsla::mpi::Matrix::page_rank_opticom(int maxIter, double beta, double epsilon, int &nb_iterations_done)
{
    //std::cout << "[PageRank] Entering PageRank" << std::endl;
    int my_mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);

    /*---- Filling local MatrixBlock data ----*/
    //std::cout << "[PageRank] Filling data for local Matrix Block" << std::endl;
    int indl, indc, pr_result_redistribution_root, result_vector_calculation_group, indl_in_result_vector_calculation_group, indc_in_result_vector_calculation_group, inter_result_vector_need_group_communicaton_group, my_result_vector_calculation_group_rank;
    long dim_l, dim_c, startRow, startColumn, endRow, endColumn, local_result_vector_size, startColumn_in_result_vector_calculation_group, startRow_in_result_vector_calculation_group;

    int pgcd_nbr_nbc, local_result_vector_size_row_blocks, local_result_vector_size_column_blocks;
    double grid_dim_factor;

    int tmp_r = this->NR, tmp_c = this->NC;
    while (tmp_c!=0) {pgcd_nbr_nbc = tmp_r % tmp_c; tmp_r = tmp_c; tmp_c = pgcd_nbr_nbc;}
    pgcd_nbr_nbc = tmp_r;

    /*this->NR = nb_blocks_row, this->NC = nb_blocks_column; this->n_row = n (dimension globale)*/
    indl = my_mpi_rank / this->NC; //indice de ligne dans la grille 2D de processus
    indc = my_mpi_rank % this->NC; //indice de colonne dans la grille 2D de processus
    dim_l = this->n_row/this->NR; //nombre de lignes dans un block
    dim_c = this->n_row/this->NC; //nombre de colonnes dans un block
    startRow = indl*dim_l;
    endRow = (indl+1)*dim_l -1;
    startColumn = indc*dim_c;
    endColumn = (indc+1)*dim_c -1;
    grid_dim_factor = (double) this->NC / (double) this->NR;
    pr_result_redistribution_root = (int) indc / grid_dim_factor;
    local_result_vector_size_column_blocks = this->NC / pgcd_nbr_nbc;
    local_result_vector_size_row_blocks = this->NR / pgcd_nbr_nbc;
    local_result_vector_size = local_result_vector_size_row_blocks * dim_l;
    result_vector_calculation_group = indl / local_result_vector_size_row_blocks;
    indl_in_result_vector_calculation_group = indl % local_result_vector_size_row_blocks;
    indc_in_result_vector_calculation_group = indc;
    inter_result_vector_need_group_communicaton_group = (indc % local_result_vector_size_column_blocks) * this->NR + indl;
    startColumn_in_result_vector_calculation_group = dim_c * (indc % local_result_vector_size_column_blocks);
    startRow_in_result_vector_calculation_group = dim_l * indl_in_result_vector_calculation_group;
    my_result_vector_calculation_group_rank = indl_in_result_vector_calculation_group * local_result_vector_size_row_blocks + indc;
    /*---- Filled MatrixBlock data ----*/

    double start_pagerank_time, total_pagerank_time;
    
    //std::cout << "[PageRank] Splitting Communicators" << std::endl;
    /* Row and Column MPI communicators */
    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, indl, indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, indc, indl, &COLUMN_COMM);

    /* Calculation group and Need group communicators */
    MPI_Comm RV_CALC_GROUP_COMM; //communicateur interne des groupes (qui regroupe sur les colonnes les blocks du même groupe de calcul)
    MPI_Comm_split(MPI_COMM_WORLD, result_vector_calculation_group, my_result_vector_calculation_group_rank, &RV_CALC_GROUP_COMM);

    MPI_Comm INTER_RV_NEED_GROUP_COMM; //communicateur externe des groupes de besoin (groupes sur les lignes) ; permet de calculer l'erreur et la somme totale du vecteur
    MPI_Comm_split(MPI_COMM_WORLD, inter_result_vector_need_group_communicaton_group, my_mpi_rank, &INTER_RV_NEED_GROUP_COMM);

    long i,j; //loops
    long cpt_iterations;
    double error_vect,error_vect_local;
    double *morceau_new_q, *morceau_new_q_local, *morceau_old_q,*tmp;
    double to_add,sum_totale_old_q,sum_totale_new_q,sum_new_q,tmp_sum;

    //init variables PageRank
    cpt_iterations = 0; error_vect=10000;//INFINITY;

    //memory allocation for old_q and new_q, and new_q initialization
    //std::cout << "[PageRank] Memory allocation for 3 vectors of size " << local_result_vector_size << std::endl;
    morceau_new_q = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_new_q_local = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_old_q = (double *)malloc(local_result_vector_size * sizeof(double));
    for (i=0;i<local_result_vector_size;i++) {morceau_new_q[i] = (double) 1/this->n_row/*this.[dimension globale]*/;}
    sum_totale_new_q = 1/*pas this->n_row = this.[dimension globale]*/;

    MPI_Barrier(MPI_COMM_WORLD);
    //start_pagerank_time = my_gettimeofday(); //Start of time measurement for PageRank

    /****************************************************************************************************/
    /****************************************** PAGERANK START ******************************************/
    /****************************************************************************************************/
    while (error_vect > epsilon /*&& !one_in_vector(morceau_new_q,local_result_vector_size)*/ && cpt_iterations<maxIter)
    {
        /************ Preparation for iteration ************/
        //old_q <=> new_q  &   sum_totale_old_q <=> sum_totale_new_q
        tmp = morceau_new_q;
        morceau_new_q = morceau_old_q;
        morceau_old_q = tmp;
        tmp_sum = sum_totale_new_q;
        sum_totale_new_q = sum_totale_old_q;
        sum_totale_old_q = tmp_sum;
        //iterations are done on new_q

        //reset morceau_new_q_local for new iteration
        for (i=0; i<local_result_vector_size; i++)
        {
            morceau_new_q_local[i] = 0;
        }

        /************ Matrix-vector product ************/
        this->Ax(&(morceau_new_q_local[startRow_in_result_vector_calculation_group]), morceau_old_q, 0);
        //this->Ax_Local_nico(morceau_new_q_local, morceau_old_q, nnz_columns_global, startColumn, startColumn_in_result_vector_calculation_group, startRow_in_result_vector_calculation_group);

        //Global Matrix-vector product new_q = P * old_q (Reduce)
        MPI_Allreduce(morceau_new_q_local, morceau_new_q, local_result_vector_size, MPI_DOUBLE, MPI_SUM, RV_CALC_GROUP_COMM); //Produit matrice_vecteur global : Reduce des morceaux de new_q dans tout les processus du même groupe de calcul
        MPI_Barrier(MPI_COMM_WORLD);

        /************ Damping ************/
        //Multiplication of the result vector by the damping factor beta and addition of norm(old_q) * (1-beta) / n
        to_add = sum_totale_old_q * (1-beta)/this->n_row/*this.[dimension globale]*/; //Ce qu'il y a à ajouter au résultat P.olq_q * beta. sum_total_old_q contient déjà la somme des éléments de old_q
        for (i=startColumn_in_result_vector_calculation_group; i<startColumn_in_result_vector_calculation_group+dim_c; i++)
        {
            morceau_new_q_local[i] = morceau_new_q_local[i] * beta + to_add; //au final new_q = beta * P.old_q + norme(old_q) * (1-beta) / n    (la partie droite du + étant ajoutée à l'initialisation)
        }

        /************ Redistribution ************/
        MPI_Bcast(morceau_new_q, local_result_vector_size, MPI_DOUBLE, pr_result_redistribution_root, COLUMN_COMM); //chaque processus d'une "ligne de processus" (dans la grille) contient le même morceau de new_q
        MPI_Barrier(MPI_COMM_WORLD);

        /************ Normalization of the new result vector ************/
        sum_new_q = 0;
        for (i=0;i<local_result_vector_size;i++) {sum_new_q += morceau_new_q[i];}
        MPI_Allreduce(&sum_new_q, &sum_totale_new_q, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes de tout les sum_new_q dans sum_totale_new_q, utile pour l'itération suivante
        for (i=0;i<local_result_vector_size;i++) {morceau_new_q[i] *= 1/sum_totale_new_q;} //normalisation avec sum totale (tout processus confondu)
	//std::cout << "[PageRank] sum_totale_new_q = " << sum_totale_new_q << std::endl;

        /************ End of iteration Operations ************/
        cpt_iterations++;
        error_vect_local = abs_two_vector_error(morceau_new_q,morceau_old_q,local_result_vector_size); //calcul de l'erreur local
        MPI_Allreduce(&error_vect_local, &error_vect, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);
    }
    /****************************************************************************************************/
    /******************************************* PAGERANK END *******************************************/
    /****************************************************************************************************/
    //cpt_iterations contains the number of iterations done, morceau_new_q are the pieces of the vector containing the PageRank

    MPI_Barrier(MPI_COMM_WORLD);
    //total_pagerank_time = my_gettimeofday() - start_pagerank_time; //end of PageRank time measurement
    nb_iterations_done = cpt_iterations;

    delete[] morceau_new_q_local;
    delete[] morceau_old_q;
    MPI_Comm_free(&ROW_COMM);
    MPI_Comm_free(&COLUMN_COMM);
    MPI_Comm_free(&RV_CALC_GROUP_COMM);
    MPI_Comm_free(&INTER_RV_NEED_GROUP_COMM);

    return morceau_new_q;
}



double * tbsla::mpi::Matrix::conjugate_gradient_opticom(int maxIter, double beta, double epsilon, int &nb_iterations_done)
{
    
    //std::cout << "[PageRank] Entering PageRank" << std::endl;
    int my_mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);

    /*---- Filling local MatrixBlock data ----*/
    //std::cout << "[PageRank] Filling data for local Matrix Block" << std::endl;
    int indl, indc, pr_result_redistribution_root, result_vector_calculation_group, indl_in_result_vector_calculation_group, indc_in_result_vector_calculation_group, inter_result_vector_need_group_communicaton_group, my_result_vector_calculation_group_rank;
    long dim_l, dim_c, startRow, startColumn, endRow, endColumn, local_result_vector_size, startColumn_in_result_vector_calculation_group, startRow_in_result_vector_calculation_group;

    int pgcd_nbr_nbc, local_result_vector_size_row_blocks, local_result_vector_size_column_blocks;
    double grid_dim_factor;

    int tmp_r = this->NR, tmp_c = this->NC;
    while (tmp_c!=0) {pgcd_nbr_nbc = tmp_r % tmp_c; tmp_r = tmp_c; tmp_c = pgcd_nbr_nbc;}
    pgcd_nbr_nbc = tmp_r;

    /*this->NR = nb_blocks_row, this->NC = nb_blocks_column; this->n_row = n (dimension globale)*/
    indl = my_mpi_rank / this->NC; //indice de ligne dans la grille 2D de processus
    indc = my_mpi_rank % this->NC; //indice de colonne dans la grille 2D de processus
    dim_l = this->n_row/this->NR; //nombre de lignes dans un block
    dim_c = this->n_row/this->NC; //nombre de colonnes dans un block
    startRow = indl*dim_l;
    endRow = (indl+1)*dim_l -1;
    startColumn = indc*dim_c;
    endColumn = (indc+1)*dim_c -1;
    grid_dim_factor = (double) this->NC / (double) this->NR;
    pr_result_redistribution_root = (int) indc / grid_dim_factor;
    local_result_vector_size_column_blocks = this->NC / pgcd_nbr_nbc;
    local_result_vector_size_row_blocks = this->NR / pgcd_nbr_nbc;
    local_result_vector_size = local_result_vector_size_row_blocks * dim_l;
    result_vector_calculation_group = indl / local_result_vector_size_row_blocks;
    indl_in_result_vector_calculation_group = indl % local_result_vector_size_row_blocks;
    indc_in_result_vector_calculation_group = indc;
    inter_result_vector_need_group_communicaton_group = (indc % local_result_vector_size_column_blocks) * this->NR + indl;
    startColumn_in_result_vector_calculation_group = dim_c * (indc % local_result_vector_size_column_blocks);
    startRow_in_result_vector_calculation_group = dim_l * indl_in_result_vector_calculation_group;
    my_result_vector_calculation_group_rank = indl_in_result_vector_calculation_group * local_result_vector_size_row_blocks + indc;
    /*---- Filled MatrixBlock data ----*/

    double start_pagerank_time, total_pagerank_time;
    
    //std::cout << "[PageRank] Splitting Communicators" << std::endl;
    /* Row and Column MPI communicators */
    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, indl, indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, indc, indl, &COLUMN_COMM);

    /* Calculation group and Need group communicators */
    MPI_Comm RV_CALC_GROUP_COMM; //communicateur interne des groupes (qui regroupe sur les colonnes les blocks du même groupe de calcul)
    MPI_Comm_split(MPI_COMM_WORLD, result_vector_calculation_group, my_result_vector_calculation_group_rank, &RV_CALC_GROUP_COMM);

    MPI_Comm INTER_RV_NEED_GROUP_COMM; //communicateur externe des groupes de besoin (groupes sur les lignes) ; permet de calculer l'erreur et la somme totale du vecteur
    MPI_Comm_split(MPI_COMM_WORLD, inter_result_vector_need_group_communicaton_group, my_mpi_rank, &INTER_RV_NEED_GROUP_COMM);

    long i,j; //loops
    long cpt_iterations;
    double alpha_i,rho,new_rho,product_w_v;
    double error_vect,error_vect_local;
    double *morceau_new_y, *morceau_new_y_local, *morceau_old_y,*tmp;
    double *morceau_new_r, *morceau_old_r;
    double *morceau_v;
    double *morceau_new_w_local, *morceau_new_w;
    
    //double to_add,sum_totale_old_q,sum_totale_new_q,sum_new_q,tmp_sum;
    //init variables PageRank
    cpt_iterations = 0; error_vect=10000;//INFINITY;

    //memory allocation for old_q and new_q, and new_q initialization
    //std::cout << "[PageRank] Memory allocation for 3 vectors of size " << local_result_vector_size << std::endl;
    morceau_new_y = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_new_y_local = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_old_y = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_new_r = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_old_r = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_v = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_new_w = (double *)malloc(local_result_vector_size * sizeof(double));
    morceau_new_w_local = (double *)malloc(local_result_vector_size * sizeof(double));
    
    for (i=0;i<local_result_vector_size;i++) {morceau_new_y[i] = (double) 1/this->n_row/*this.[dimension globale]*/;}
    //reset morceau_new_y_local for new iteration
    for (i=0; i<local_result_vector_size; i++)
    {
        morceau_new_y_local[i] = 0;
    }
    
    /**************************************************************************************************************************/
    /******************************************** CONJUGATE GRADIENT initialization ********************************************/
    /**************************************************************************************************************************/
    
    
    this->Ax(&(morceau_new_y_local[startRow_in_result_vector_calculation_group]), morceau_new_y, 0);
    MPI_Allreduce(morceau_new_y_local, morceau_new_r, local_result_vector_size, MPI_DOUBLE, MPI_SUM, RV_CALC_GROUP_COMM); //Produit matrice_vecteur global : Reduce des morceaux de new_q dans tout les processus du même groupe de calcul
    MPI_Barrier(MPI_COMM_WORLD);
    double somme_b=0;
    double morceau_b[local_result_vector_size];
    for(i=0;i<local_result_vector_size;i++){
        morceau_b[i]=rand();
    }
    MPI_Bcast(morceau_b, local_result_vector_size, MPI_DOUBLE, 0, COLUMN_COMM); //chaque processus d'une "ligne de processus" (dans la grille) contient le même morceau de new_r
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(i=0;i<local_result_vector_size;i++){
        somme_b+=morceau_b[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &new_rho, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale

    for(i=0;i<local_result_vector_size;i++){
        morceau_b[i]=morceau_b[i]/somme_b;
    }
    
    for(i=0;i<local_result_vector_size;i++){
        morceau_new_r[i]=morceau_b[i]-morceau_new_r[i];
    }
    
    
    /************* p0=(r0,r0) *************/
    for (i=startColumn_in_result_vector_calculation_group; i<startColumn_in_result_vector_calculation_group+dim_c; i++)
    {
        new_rho+=morceau_new_r[i]*morceau_new_r[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &new_rho, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
    MPI_Barrier(MPI_COMM_WORLD);
    
    /************* V0=r0 *************/
    for(i=0;i<local_result_vector_size;i++){
        morceau_v[i]=morceau_new_r[i];
    }
    /*std::cout<<"vector b:";
    for (i=0;i<local_result_vector_size;i++){
    	std::cout<<morceau_b[i]<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"morceau_r debut:";
    for(i=0;i<local_result_vector_size;i++){
        std::cout<<morceau_new_r<<" ";
    }
    std::cout<<std::endl;
    */
    /************************************************************************************************************/
    /***************************************** CONJUGATE GRADIENT START *****************************************/
    /************************************************************************************************************/
    while (error_vect > epsilon /*&& !one_in_vector(morceau_new_q,local_result_vector_size)*/ && cpt_iterations<maxIter)
    {
        /************ Preparation for iteration ************/
        //old_v <=> new_v  &   old_r <=> new_r & old_v <=> new_v
        tmp = morceau_new_r;
        morceau_new_r = morceau_old_r;
        morceau_old_r = tmp;
        tmp = morceau_new_y;
        morceau_new_y = morceau_old_y;
        morceau_old_y = tmp;
        rho=new_rho;
        
        //reset product_w_v, new_rho for new iteration
        product_w_v=0;
        new_rho=0;
        //iterations are done on new_v

        //reset morceau_new_w_local for new iteration
        for (i=0; i<local_result_vector_size; i++)
        {
            morceau_new_w_local[i] = 0;
        }
        
        /************* w_i=A_i*vi *************/
        this->Ax(&(morceau_new_w_local[startRow_in_result_vector_calculation_group]), morceau_v, 0);
        MPI_Allreduce(morceau_new_w_local, morceau_new_w, local_result_vector_size, MPI_DOUBLE, MPI_SUM, RV_CALC_GROUP_COMM); //Produit matrice_vecteur global : Reduce des morceaux de new_q dans tout les processus du même groupe de calcul
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(morceau_new_w, local_result_vector_size, MPI_DOUBLE, pr_result_redistribution_root, COLUMN_COMM); //chaque processus d'une "ligne de processus" (dans la grille) contient le même morceau de new_w
        MPI_Barrier(MPI_COMM_WORLD);
        
        /************* alpha_i= p_i/(w_i,v_i) *************/
        for(i=0;i<local_result_vector_size;i++){
            product_w_v+=morceau_new_w[i]*morceau_v[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &product_w_v, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);
        alpha_i=rho/product_w_v;
        
        /************* y_i+1= y_i+alpha_i*v_i *************/
        for (i=0; i<local_result_vector_size; i++)
        {
            morceau_new_y[i] = morceau_v[i] * alpha_i + morceau_old_y[i]; 
        }
        
        /************* r_i+1= r_i+alpha_i*w_i *************/
        for (i=0; i<local_result_vector_size; i++)
        {
            morceau_new_r[i] = morceau_old_r[i]-morceau_new_w[i] * alpha_i ; 
        }
        
        /************* p_i+1= (r_i+1,r_i+1) *************/
        for (i=0; i<local_result_vector_size; i++)
        {
            new_rho += morceau_new_r[i] * morceau_new_r[i]; 
        }
        MPI_Allreduce(MPI_IN_PLACE, &new_rho, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);
        /************* beta_i= p_i+1/p_i *************/
        beta=new_rho/rho;
        /************* v_i+1= r_i+1+beta_i*v_i *************/
        for (i=0; i<local_result_vector_size; i++)
        {
            morceau_v[i]=morceau_new_r[i]+beta*morceau_v[i];
        }
        
        /************ End of iteration Operations ************/
        cpt_iterations++;
        error_vect_local = abs_one_vector_error(morceau_new_r,local_result_vector_size); //calcul de l'erreur local
        MPI_Allreduce(&error_vect_local, &error_vect, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "iteration: "<<cpt_iterations<<", error local: "<<error_vect_local <<", error vect:"<<error_vect<<std::endl;
    }
    /****************************************************************************************************/
    /******************************************* CONJUGATE GRADIENT END *******************************************/
    /****************************************************************************************************/
    //cpt_iterations contains the number of iterations done, morceau_new_q are the pieces of the vector containing the PageRank
    /*std::cout<<"morceau_r fin:";
    for(i=0;i<local_result_vector_size;i++){
    	std::cout<<morceau_new_r<<" ";
    }
    std::cout<<std::endl;
*/

    MPI_Barrier(MPI_COMM_WORLD);
    //total_pagerank_time = my_gettimeofday() - start_pagerank_time; //end of PageRank time measurement
    nb_iterations_done = cpt_iterations;

    delete[] morceau_v;
    delete[] morceau_new_w_local;
    delete[] morceau_new_w;
    delete[] morceau_new_r;
    delete[] morceau_old_r;
    delete[] morceau_new_y_local;
    delete[] morceau_old_y;
    MPI_Comm_free(&ROW_COMM);
    MPI_Comm_free(&COLUMN_COMM);
    MPI_Comm_free(&RV_CALC_GROUP_COMM);
    MPI_Comm_free(&INTER_RV_NEED_GROUP_COMM);

    return morceau_new_y;
}
