#ifndef SOLVERS_H
#define SOLVERS_H

#include <complex.h>

// Макрос для линейной индексации двумерных массивов (Row-Major Order)
#define IDX(row, col, cols) ((row) * (cols) + (col))

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL
#endif

// =====================================================================
// ЯДРА УМНОЖЕНИЯ МАТРИЦ
// =====================================================================

void matmul_naive(int n, int m, int p, const double complex *A, const double complex *B, double complex *C, double complex *B_T);
void matmul_perminov(int n, int k_dim, int m, const double complex *A, const double complex *B, double complex *C, double complex *B_T);
void matmul_winograd(int n, int m_dim, int p, const double complex *A, const double complex *B, double complex *C, double complex *row_f, double complex *col_f, double complex *B_T);

// =====================================================================
// ВНУТРЕННИЕ РЕШАТЕЛИ СИСТЕМ (Прямой/обратный ход)
// =====================================================================

void gauss_solve_inplace(int n, double complex *Ah_A, const double complex *Ah_y, double complex *w, double complex *M);
void cholesky_solve_inplace(int n, double complex *Ah_A, const double complex *Ah_y, double complex *w);

// =====================================================================
// ОСНОВНОЙ API СОЛВЕРОВ ДЛЯ БЕНЧМАРКА И СВЯЗКИ С PYTHON
// =====================================================================

EXPORT void solve_gauss_naive(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *M, double complex *BT);
EXPORT void solve_cholesky_naive(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *BT);

EXPORT void solve_gauss_perminov(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *M, double *tA, double complex *BT, double *tBT);
EXPORT void solve_cholesky_perminov(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double *tA, double complex *BT, double *tBT);

EXPORT void solve_gauss_winograd(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *M, double complex *rf, double complex *cf, double complex *BT);
EXPORT void solve_cholesky_winograd(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *rf, double complex *cf, double complex *BT);

EXPORT void solve_qr_householder(int n_s, int n_f, int b_s, const double complex *A, const double complex *y, double reg, double complex *w, double complex *R, double complex *y_ext, double complex *v);

#endif // SOLVERS_H