/**
 * @file
 * @brief Vector parameter wrapper for XMP
 *
 *
 *
 * 2020-06-08
 *
 */
#ifndef VECTOR_XMP_TYPE_HH
#define VECTOR_XMP_TYPE_HH 1

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <mpi.h>
#include <tbsla/cpp/Vector.h>
#include <tbsla/cpp/utils/range.h>

#define MAX_SIZE 1024

struct Vector_struct {
  int n_row, n_col; // global matrix size
  int ln_row, ln_col; // local matrix size
  int bn_row, bn_col; // block matrix size
  int pr, pc, gr, gc; // positionning in the fine grain grid
  int bpr, bpc, bgr, bgc; // positionning in the coarse grain grid
  int lpr, lpc, lgr, lgc; // positionning in the task fine grain grid
  C_CPP_Vector_t *v;
};

typedef struct Vector_struct XMP_Vector; /* Declaration of parameter type in XMP ( XMP_type )*/
typedef struct Vector_struct* Vector; /* Declaration of parameter type for import/export functions (type) */

static MPI_Datatype Vector_MPI_Type()
{
  return MPI_DOUBLE;
}
// param_import / export definition for types that need data distribution in XMP

static bool Vector_import(Vector param, char* filename, const MPI_Datatype motif, const int size)
{
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char new_filename[MAX_SIZE];
  sprintf(new_filename, "%s___%d", filename, rank);
  FILE *fp;
  fp = fopen(new_filename, "rb");
  int r = 0;
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
  if (r < 18)
    return false;
  fclose(fp);
  int seek = 18 * sizeof(int);
  param->v = C_CPP_Vector_create();
  return C_CPP_Vector_read(param->v, new_filename, seek);
}

static bool Vector_export(Vector param, char* filename, const MPI_Datatype motif, const int size, MPI_Comm Communicator)
{
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char new_filename[MAX_SIZE];
  sprintf(new_filename, "%s___%d", filename, rank);
  int r;
  FILE *fp;
  fp = fopen(new_filename, "wb");
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
  if (r < 18)
    return false;
  return C_CPP_Vector_write(param->v, new_filename);
}

#endif
