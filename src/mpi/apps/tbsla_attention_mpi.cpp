#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

static std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

void distribute_dense_matrix(const double* B, double* B_local, int rows_B, int cols_B, int ln_rows, MPI_Comm comm) {
    // Scatter dense matrix rows to processes
    MPI_Scatter(B, ln_rows * cols_B, MPI_DOUBLE, B_local, ln_rows * cols_B, MPI_DOUBLE, 0, comm);
}

double compute_gflops(double runtime, int nnz, int cols_B) {
    return (2.0 * nnz * cols_B) / (runtime * 1e9); // GFLOPS
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    InputParser input(argc, argv);

    // Input parsing
    std::string matrix_dim_string = input.get_opt("--matrix_dim", "1024");
    std::string gr_string = input.get_opt("--GR", "1");
    std::string gc_string = input.get_opt("--GC", "1");
    std::string nnz_ratio_string = input.get_opt("--NNZ", "0.01");
    std::string cols_B_string = input.get_opt("--cols_B", "512");

    int matrix_dim = std::stoi(matrix_dim_string);
    int GR = std::stoi(gr_string);
    int GC = std::stoi(gc_string);
    double nnz_ratio = std::stod(nnz_ratio_string);
    int cols_B = std::stoi(cols_B_string);

    if (GR * GC != world) {
        if (rank == 0) {
            std::cerr << "Grid dimensions (GR x GC) do not match the number of processes!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Process grid position
    int pr = rank / GC;
    int pc = rank % GC;

    // Local matrix dimensions
    int ln_row = matrix_dim / GR; // Rows in the local block
    int ln_col = matrix_dim / GC; // Columns in the local block

    // Initialize sparse matrix (CSR format)
    tbsla::mpi::MatrixCSR A;
    A.fill_random(matrix_dim, matrix_dim, nnz_ratio, rank / GC, rank % GC, GR, GC);

    // Initialize dense matrix B (only on root)
    double* B = nullptr;
    if (rank == 0) {
        B = new double[matrix_dim * cols_B];
        std::fill(B, B + matrix_dim * cols_B, 1.0); // Fill with dummy data
    }

    // Local dense matrix block
    double* B_local = new double[ln_row * cols_B];
    distribute_dense_matrix(B, B_local, matrix_dim, cols_B, ln_row, MPI_COMM_WORLD);

    // Local result matrix
    double* C_local = new double[ln_row * cols_B];
    std::memset(C_local, 0, sizeof(double) * ln_row * cols_B);

    // Perform sparse-dense multiplication
    auto t_start = now();
    A.dense_multiply(B_local, C_local, cols_B, MPI_COMM_WORLD);
    auto t_end = now();

    // Collect and output results
    double runtime = (t_end - t_start) / 1e9; // in seconds
    long int nnz = A.compute_sum_nnz(MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = compute_gflops(runtime, nnz, cols_B);
        std::cout << "Matrix Dimension: " << matrix_dim << " x " << matrix_dim << std::endl;
        std::cout << "Dense Matrix Columns: " << cols_B << std::endl;
        std::cout << "Runtime: " << runtime << " seconds" << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;
        std::cout << "NNZ: " << nnz << std::endl;
    }

    // Clean up
    delete[] B_local;
    delete[] C_local;
    if (rank == 0) delete[] B;

    MPI_Finalize();
    return 0;
}
