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
    int rows_per_block = rows_B / p;
    int pr = rank / p;
    int pc = rank % p;
    
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &row_comm);
    
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

    MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE, B_local, ln_rows_B * cols_B, MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        delete[] sendcounts;
        delete[] displs;
    }
}

// Compute GFLOPS for sparse-dense multiplication
double compute_gflops(double runtime, int nnz, int cols_B) {
    return (2.0 * nnz * cols_B) / (runtime * 1e9);
}

void print_dense_matrix(double* M, int nb_row, int nb_col) {
    for (int i = 0; i < nb_row; i++) {
        for (int j = 0; j < nb_col; j++) {
            std::cout << M[i * nb_col + j];
        }
        std::cout << std::endl;
    }
}

void fill_matrix_by_blocks(double* B, int matrix_dim, int cols_B, int n_blocks) {
    int rows_per_block = matrix_dim / n_blocks;
    int extra_rows = matrix_dim % n_blocks;
    int current_row = 0;

    for (int block_id = 0; block_id < n_blocks; ++block_id) {
        int block_rows = rows_per_block + (block_id < extra_rows ? 1 : 0);
        for (int i = 0; i < block_rows; ++i) {
            for (int j = 0; j < cols_B; ++j) {
                B[current_row * cols_B + j] = 1.0 + static_cast<double>(block_id);
            }
            ++current_row;
        }
    }
}

void debug_print(int rank, int world, double* B_local, double* C_local, tbsla::mpi::Matrix* m, int ln_row, int cols_B) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world; ++i) {
        if (rank == i) {
            std::cout << "=== Debugging Rank " << rank << " ===" << std::endl;
            std::cout << "=== End of Rank " << rank << " ===" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int p = std::sqrt(world);
    if (p * p != world) {
        if (rank == 0) {
            std::cerr << "The number of processes must be a perfect square!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    InputParser input(argc, argv);

    std::string matrix_dim_string = input.get_opt("--matrix_dim", "1024");
    std::string nnz_ratio_string = input.get_opt("--NNZ", "0.01");
    std::string cols_B_string = input.get_opt("--cols_B", "512");
    std::string base_string = input.get_opt("--base", "-1");
    bool skip_softmax = input.has_opt("--skip_softmax");
    bool skip_multiplication = input.has_opt("--skip_multiplication");

    int matrix_dim = std::stoi(matrix_dim_string);
    double nnz_ratio = std::stod(nnz_ratio_string);
    int cols_B = std::stoi(cols_B_string);
    int base = std::stoi(base_string);

    int pr = rank / p;
    int pc = rank % p;
    
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &row_comm);

    int ln_row_A = matrix_dim / p;
    int ln_col_A = matrix_dim / p;
    int ln_rows_B = matrix_dim / p;

    tbsla::mpi::Matrix* m;
    m = new tbsla::mpi::MatrixCSR();

    auto t_init_start = now();
    m->fill_random(matrix_dim, matrix_dim, nnz_ratio, 0, pr, pc, p, p);
    auto t_init_end = now();

    double* max_abs = nullptr;
    double* s = new double[m->get_ln_row()];

    auto csr_matrix = dynamic_cast<tbsla::mpi::MatrixCSR*>(m);

    double t_op_start = 0, t_op_end = 0;
    if (csr_matrix && !skip_softmax) {
        t_op_start = now();
        if (input.has_opt("--with_max-abs")) {
            max_abs = new double[m->get_ln_row()];
            MPI_Barrier(MPI_COMM_WORLD);
            csr_matrix->get_row_max_abs(max_abs);
            csr_matrix->reduce_row_max_abs(MPI_COMM_WORLD, max_abs);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        csr_matrix->apply_exponential(max_abs, base);

        MPI_Barrier(MPI_COMM_WORLD);
        m->get_row_sums(s);
        csr_matrix->reduce_row_sums(MPI_COMM_WORLD, s);
        MPI_Barrier(MPI_COMM_WORLD);

        m->normalize_rows(s);

        t_op_end = now();
        std::cout << "Time softmax = " << std::to_string((t_op_end - t_op_start) / 1e9) << std::endl;
    } else if (!csr_matrix) {
        std::cerr << "Error: m is not of type MatrixCSR!" << std::endl;
    }

    delete[] s;
    if (input.has_opt("--with_max-abs")) delete[] max_abs;

    double* B = nullptr;
    if (rank == 0) {
        B = new double[matrix_dim * cols_B];
        fill_matrix_by_blocks(B, matrix_dim, cols_B, p);
    }

    double* B_local = new double[ln_rows_B * cols_B];
    auto t_distribute_start = now();
    distribute_dense_matrix(B, B_local, matrix_dim, cols_B, ln_rows_B, p, rank, MPI_COMM_WORLD);
    auto t_distribute_end = now();

    double* C_local = new double[ln_row_A * cols_B];
    std::memset(C_local, 0, sizeof(double) * ln_row_A * cols_B);

    double t_multiply_start = 0, t_multiply_end = 0;
    if (!skip_multiplication) {
        t_multiply_start = now();
        m->dense_multiply(B_local, C_local, cols_B, MPI_COMM_WORLD);
        m->row_sum_reduction(C_local, ln_row_A, cols_B, row_comm);
        t_multiply_end = now();
    }

    debug_print(rank, world, B_local, C_local, m, ln_row_A, cols_B);

    delete[] B_local;
    delete[] C_local;
    MPI_Comm_free(&row_comm);  
    if (rank == 0) delete[] B;

    auto t_finalize_start = now();
    MPI_Finalize();
    auto t_finalize_end = now();

    if (rank == 0) {
        // Manually construct JSON string
        std::string json_output = "{";
        json_output += "\"parameters\": {";
        json_output += "\"matrix_dim\": " + std::to_string(matrix_dim) + ",";
        json_output += "\"nnz_ratio\": " + std::to_string(nnz_ratio) + ",";
        json_output += "\"cols_B\": " + std::to_string(cols_B) + ",";
        json_output += "\"base\": " + std::to_string(base) + ",";
        json_output += "\"world_size\": " + std::to_string(world);
        json_output += "},";
        json_output += "\"timings\": {";
        json_output += "\"initialization\": " + std::to_string((t_init_end - t_init_start) / 1e9) + ",";
        json_output += "\"softmax_operations\": " + (skip_softmax ? "0" : std::to_string((t_op_end - t_op_start) / 1e9)) + ",";
        json_output += "\"matrix_distribution\": " + std::to_string((t_distribute_end - t_distribute_start) / 1e9) + ",";
        json_output += "\"multiplication\": " + (skip_multiplication ? "0" : std::to_string((t_multiply_end - t_multiply_start) / 1e9)) + ",";
        json_output += "\"finalization\": " + std::to_string((t_finalize_end - t_finalize_start) / 1e9);
        json_output += "}";
        json_output += "}";

        std::cout << json_output << std::endl;
    }

    return 0;
}