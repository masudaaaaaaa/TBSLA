/**
 * @file
 * @brief Sparse Matrix parameter wrapper for XMP
 *
 *
 *
 * 2020-06-02
 *
 */
#ifndef SPARSEMATRIX_XMP_TYPE_HH
#define SPARSEMATRIX_XMP_TYPE_HH 1

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <mpi.h>
#include <tbsla/cpp/MatrixCOO.h>
#include <tbsla/cpp/MatrixSCOO.h>
#include <tbsla/cpp/MatrixCSR.h>
#include <tbsla/cpp/MatrixELL.h>
#include <tbsla/cpp/MatrixDENSE.h>
#include <tbsla/cpp/utils/range.h>
#include <tbsla/mpi/vector_comm.h>

#define MATRIX_FORMAT_COO 1
#define MATRIX_FORMAT_SCOO 2
#define MATRIX_FORMAT_CSR 3
#define MATRIX_FORMAT_ELL 4
#define MATRIX_FORMAT_DENSE 5

#define MAX_SIZE 1024

struct SparseMatrix_struct {
  int n_row, n_col; // global matrix size
  int ln_row, ln_col; // local matrix size
  int bn_row, bn_col; // block matrix size
  int pr, pc, gr, gc; // positionning in the fine grain grid
  int bpr, bpc, bgr, bgc; // positionning in the coarse grain grid
  int lpr, lpc, lgr, lgc; // positionning in the task fine grain grid
  int matrixformat;
  C_CPP_MatrixCOO_t *mcoo;
  C_CPP_MatrixSCOO_t *mscoo;
  C_CPP_MatrixCSR_t *mcsr;
  C_CPP_MatrixELL_t *mell;
  C_CPP_MatrixDENSE_t *mdense;
};

typedef struct SparseMatrix_struct XMP_SparseMatrix; /* Declaration of parameter type in XMP ( XMP_type )*/
typedef struct SparseMatrix_struct* SparseMatrix; /* Declaration of parameter type for import/export functions (type) */

static MPI_Datatype SparseMatrix_MPI_Type()
{
  return MPI_DOUBLE;
}
// param_import / export definition for types that need data distribution in XMP

static bool SparseMatrix_import(SparseMatrix param, char* filename, const MPI_Datatype motif, const int size)
{
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char new_filename[MAX_SIZE];
  sprintf(new_filename, "%s___%d", filename, rank);
  FILE *fp;
  fp = fopen(new_filename, "rb");
  int r = 0;
  r += fread(&(param->matrixformat), sizeof(int), 1, fp);
  r += fread(&(param->n_row), sizeof(int), 1, fp);
  r += fread(&(param->n_col), sizeof(int), 1, fp);
  r += fread(&(param->ln_row), sizeof(int), 1, fp);
  r += fread(&(param->ln_col), sizeof(int), 1, fp);
  r += fread(&(param->bn_row), sizeof(int), 1, fp);
  r += fread(&(param->bn_col), sizeof(int), 1, fp);
  r += fread(&(param->pr), sizeof(int), 1, fp);
  r += fread(&(param->pc), sizeof(int), 1, fp);
  r += fread(&(param->gr), sizeof(int), 1, fp);
  r += fread(&(param->gc), sizeof(int), 1, fp);
  r += fread(&(param->bpr), sizeof(int), 1, fp);
  r += fread(&(param->bpc), sizeof(int), 1, fp);
  r += fread(&(param->bgr), sizeof(int), 1, fp);
  r += fread(&(param->bgc), sizeof(int), 1, fp);
  r += fread(&(param->lpr), sizeof(int), 1, fp);
  r += fread(&(param->lpc), sizeof(int), 1, fp);
  r += fread(&(param->lgr), sizeof(int), 1, fp);
  r += fread(&(param->lgc), sizeof(int), 1, fp);
  if (r < 19)
    return false;
  fclose(fp);
  int seek = 19 * sizeof(int);
  if(param->matrixformat == MATRIX_FORMAT_COO) {
    param->mcoo = C_CPP_MatrixCOO_create();
    bool r = C_CPP_MatrixCOO_read(param->mcoo, new_filename, seek);
    return r;
  } else if(param->matrixformat == MATRIX_FORMAT_SCOO) {
    param->mscoo = C_CPP_MatrixSCOO_create();
    bool r = C_CPP_MatrixSCOO_read(param->mscoo, new_filename, seek);
    return r;
  } else if(param->matrixformat == MATRIX_FORMAT_CSR) {
    param->mcsr = C_CPP_MatrixCSR_create();
    bool r = C_CPP_MatrixCSR_read(param->mcsr, new_filename, seek);
    return r;
  } else if(param->matrixformat == MATRIX_FORMAT_ELL) {
    param->mell = C_CPP_MatrixELL_create();
    bool r = C_CPP_MatrixELL_read(param->mell, new_filename, seek);
    return r;
  } else if(param->matrixformat == MATRIX_FORMAT_DENSE) {
    param->mdense = C_CPP_MatrixDENSE_create();
    bool r = C_CPP_MatrixDENSE_read(param->mdense, new_filename, seek);
    return r;
  }

  return false;
}

static bool SparseMatrix_export(const SparseMatrix param, char* filename, const MPI_Datatype motif, const int size, MPI_Comm Communicator)
{
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char new_filename[MAX_SIZE];
  sprintf(new_filename, "%s___%d", filename, rank);
  FILE *fp;
  fp = fopen(new_filename, "wb");
  int r;
  r += fwrite(&(param->matrixformat), sizeof(int), 1, fp);
  r += fwrite(&(param->n_row), sizeof(int), 1, fp);
  r += fwrite(&(param->n_col), sizeof(int), 1, fp);
  r += fwrite(&(param->ln_row), sizeof(int), 1, fp);
  r += fwrite(&(param->ln_col), sizeof(int), 1, fp);
  r += fwrite(&(param->bn_row), sizeof(int), 1, fp);
  r += fwrite(&(param->bn_col), sizeof(int), 1, fp);
  r += fwrite(&(param->pr), sizeof(int), 1, fp);
  r += fwrite(&(param->pc), sizeof(int), 1, fp);
  r += fwrite(&(param->gr), sizeof(int), 1, fp);
  r += fwrite(&(param->gc), sizeof(int), 1, fp);
  r += fwrite(&(param->bpr), sizeof(int), 1, fp);
  r += fwrite(&(param->bpc), sizeof(int), 1, fp);
  r += fwrite(&(param->bgr), sizeof(int), 1, fp);
  r += fwrite(&(param->bgc), sizeof(int), 1, fp);
  r += fwrite(&(param->lpr), sizeof(int), 1, fp);
  r += fwrite(&(param->lpc), sizeof(int), 1, fp);
  r += fwrite(&(param->lgr), sizeof(int), 1, fp);
  r += fwrite(&(param->lgc), sizeof(int), 1, fp);
  fclose(fp);
  if (r < 19)
    return false;
  if(param->matrixformat == MATRIX_FORMAT_COO) {
    return C_CPP_MatrixCOO_write(param->mcoo, new_filename);
  } else if(param->matrixformat == MATRIX_FORMAT_SCOO) {
    return C_CPP_MatrixSCOO_write(param->mscoo, new_filename);
  } else if(param->matrixformat == MATRIX_FORMAT_CSR) {
    return C_CPP_MatrixCSR_write(param->mcsr, new_filename);
  } else if(param->matrixformat == MATRIX_FORMAT_ELL) {
    return C_CPP_MatrixELL_write(param->mell, new_filename);
  } else if(param->matrixformat == MATRIX_FORMAT_DENSE) {
    return C_CPP_MatrixDENSE_write(param->mdense, new_filename);
  }

  return false;
}

#endif
