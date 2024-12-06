#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random> 

// Function to measure current time in nanoseconds
static std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

  // Distribute dense matrix rows across MPI processes
  void distribute_dense_matrix(const double* B, double* B_local, int rows_B, int cols_B, int ln_rows_B, int p, int rank, MPI_Comm comm) {
      // Compute the number of rows in each block
      int rows_per_block = rows_B / p;
  
      // Rank's process grid row and column position
      int pr = rank / p; // Row in the process grid
      int pc = rank % p; // Column in the process grid
  
      // Buffer for scatterv
      int* sendcounts = nullptr;
      int* displs = nullptr;
  
      if (rank == 0) {
          sendcounts = new int[p * p];
          displs = new int[p * p];
          for (int i = 0; i < p * p; ++i) {
              sendcounts[i] = rows_per_block * cols_B;
              displs[i] = (i % p) * rows_per_block * cols_B;
          }
      }
  
      // Scatter rows of B along the rows of the process grid
      MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE, B_local, ln_rows_B * cols_B, MPI_DOUBLE, 0, comm);
  
      if (rank == 0) {
          delete[] sendcounts;
          delete[] displs;
      }
  }
  


// Compute GFLOPS for sparse-dense multiplication
double compute_gflops(double runtime, int nnz, int cols_B) {
    return (2.0 * nnz * cols_B) / (runtime * 1e9); // GFLOPS
}

void print_dense_matrix(double* M, int nb_row, int nb_col) {
    for (int i=0; i<nb_row; i++){
        for (int j=0; j<nb_col; j++){
            std::cout << M[i*nb_col + j];
        }
            std::cout << std::endl;
    }
}

  void fill_matrix_by_blocks(double* B, int matrix_dim, int cols_B, int n_blocks) {
      int rows_per_block = matrix_dim / n_blocks; // Rows in each block
      int extra_rows = matrix_dim % n_blocks;    // Handle remaining rows
  
      int current_row = 0; // Track the current row in B
      for (int block_id = 0; block_id < n_blocks; ++block_id) {
          int block_rows = rows_per_block + (block_id < extra_rows ? 1 : 0); // Distribute extra rows
  
          for (int i = 0; i < block_rows; ++i) {
              for (int j = 0; j < cols_B; ++j) {
                  B[current_row * cols_B + j] = 1.0 + static_cast<double>(block_id);
              }
              ++current_row;
          }
      }
  }

void debug_print(int rank, int world, double* B_local, double* C_local, tbsla::mpi::Matrix* m, int ln_row, int cols_B) {
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes reach this point
    for (int i = 0; i < world; ++i) {
        if (rank == i) {
            std::cout << "=== Debugging Rank " << rank << " ===" << std::endl;

            // Print the sparse matrix state
            std::cout << "Sparse Matrix (local to rank " << rank << "):" << std::endl;
            std::cout << *m << std::endl;

            // Print local dense matrix B
            std::cout << "Local Dense Matrix B (rank " << rank << "):" << std::endl;
            print_dense_matrix(B_local, ln_row, cols_B);

            // Print local result matrix C
            std::cout << "Local Result Matrix C (rank " << rank << "):" << std::endl;
            print_dense_matrix(C_local, ln_row, cols_B);

            std::cout << "=== End of Rank " << rank << " ===" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize before moving to the next rank
    }
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Ensure we have p^2 processes
    int p = std::sqrt(world);
    if (p * p != world) {
        if (rank == 0) {
            std::cerr << "The number of processes must be a perfect square!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    InputParser input(argc, argv);

    // Parse input arguments
    std::string matrix_dim_string = input.get_opt("--matrix_dim", "1024");
    std::string nnz_ratio_string = input.get_opt("--NNZ", "0.01");
    std::string cols_B_string = input.get_opt("--cols_B", "512");
    std::string base_string = input.get_opt("--base", "-1");

    int matrix_dim = std::stoi(matrix_dim_string);
    double nnz_ratio = std::stod(nnz_ratio_string);
    int cols_B = std::stoi(cols_B_string);
    int base = std::stoi(base_string);

    // Process grid position
    int pr = rank / p;
    int pc = rank % p;

    // Local matrix dimensions
    int ln_row_A = matrix_dim / p; // Rows in the local block of A
    int ln_col_A = matrix_dim / p; // Columns in the local block of A
    int ln_rows_B = matrix_dim / p; // Rows in the local block of B

    // Declare sparse matrix A as a pointer to the base type
    tbsla::mpi::Matrix* m;
    m = new tbsla::mpi::MatrixCSR();

    // Initialize sparse matrix A
    m->fill_random(matrix_dim, matrix_dim, nnz_ratio, 0 /*seed*/, pr, pc, p, p);

    // Perform Softmax
    
    double* max_abs = nullptr;
    double* s = new double[m->get_ln_row()];
    
    // Attempt to cast m to MatrixCSR
    auto csr_matrix = dynamic_cast<tbsla::mpi::MatrixCSR*>(m);
    
    if (csr_matrix) {
        // Only call MatrixCSR-specific methods if m is actually a MatrixCSR object
        csr_matrix->print_dense(std::cout, MPI_COMM_WORLD);
        
        auto t_op_start = now();
        // First, compute and reduce row max abs if the parameter is specified in the input
        if (input.has_opt("--with_max-abs")) {
          max_abs = new double[m->get_ln_row()];
          MPI_Barrier(MPI_COMM_WORLD);
          csr_matrix->get_row_max_abs(max_abs);
          csr_matrix->reduce_row_max_abs(MPI_COMM_WORLD, max_abs);
          MPI_Barrier(MPI_COMM_WORLD);
        }
        
        // Applying exponential
        csr_matrix->apply_exponential(max_abs, base);
        
        // Compute and reduce row sums
        MPI_Barrier(MPI_COMM_WORLD);
        m->get_row_sums(s);
        csr_matrix->reduce_row_sums(MPI_COMM_WORLD, s);
        MPI_Barrier(MPI_COMM_WORLD);

        m->normalize_rows(s);
    
        auto t_op_end = now();
        std::cout << "Time softmax = " << std::to_string((t_op_end - t_op_start) / 1e9) << std::endl;
    
        // Print the dense matrix after operations
        csr_matrix->print_dense(std::cout, MPI_COMM_WORLD);
    } else {
        std::cerr << "Error: m is not of type MatrixCSR!" << std::endl;
    }
    
    // Clean up
    delete[] s;
    if (input.has_opt("--with_max-abs")) delete[] max_abs;


    // Initialize dense matrix B (only on root process)
    double* B = nullptr;
    if (rank == 0) {
        B = new double[matrix_dim * cols_B];
        fill_matrix_by_blocks(B, matrix_dim, cols_B, p); // Divide B into p blocks
    }

    // Local dense matrix block
    double* B_local = new double[ln_rows_B * cols_B];
    distribute_dense_matrix(B, B_local, matrix_dim, cols_B, ln_rows_B, p, rank, MPI_COMM_WORLD);
    
    // Local result matrix
    double* C_local = new double[ln_row_A * cols_B];
    std::memset(C_local, 0, sizeof(double) * ln_row_A * cols_B);
    
    std::cout << "Print before multiplication" << std::endl;  
    debug_print(rank, world, B_local, C_local, m, ln_row_A, cols_B);
    
    // Perform sparse-dense multiplication
    m->dense_multiply(B_local, C_local, cols_B, MPI_COMM_WORLD);
    
    m->row_sum_reduction(C_local, ln_rows_B, cols_B, pr, pc, MPI_COMM_WORLD);
    
    // Debug print
    std::cout << "Print after multiplication" << std::endl;
    debug_print(rank, world, B_local, C_local, m, ln_row_A, cols_B);

    // Finalize
    delete[] B_local;
    delete[] C_local;
    if (rank == 0) delete[] B;

    MPI_Finalize();
    return 0;
}

