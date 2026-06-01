#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include "solvers.h" // Убедитесь, что заголовки лежат там же

void generate_dummy_data(double complex *A, double complex *y, int N_s, int N_f) {
    for (int i = 0; i < N_s * N_f; i++) {
        A[i] = ((double)rand() / RAND_MAX) + ((double)rand() / RAND_MAX) * I;
    }
    for (int i = 0; i < N_s; i++) {
        y[i] = ((double)rand() / RAND_MAX) + ((double)rand() / RAND_MAX) * I;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <solver_name> <n_features>\n", argv[0]);
        return 1;
    }

    char *solver = argv[1];
    int N_f = atoi(argv[2]);
    int N_s = 10000; 
    int b_s = 32;       
    int iterations = 200; 
    double reg = 1e-6;

    // Выделение памяти (сумма всех нужд для любого решателя)
    double complex *A = malloc(N_s * N_f * sizeof(double complex));
    double complex *y = malloc(N_s * sizeof(double complex));
    double complex *w = malloc(N_f * sizeof(double complex));
    double complex *Ah = malloc(N_f * b_s * sizeof(double complex));
    double complex *Ah_A = malloc(N_f * N_f * sizeof(double complex));
    double complex *Ah_y = malloc(N_f * sizeof(double complex));
    double complex *M = malloc(N_f * (N_f + 1) * sizeof(double complex));
    double complex *rf = malloc(N_f * sizeof(double complex));
    double complex *cf = malloc(N_f * sizeof(double complex));
    double complex *BT = malloc(b_s * N_f * sizeof(double complex));
    double *tA = malloc(N_f * b_s * sizeof(double) * 2);
    double *tBT = malloc(b_s * N_f * sizeof(double) * 2);
    double complex *R = malloc((N_s + N_f) * N_f * sizeof(double complex));
    double complex *y_ext = malloc((N_s + N_f) * sizeof(double complex));
    double complex *v = malloc((N_s + N_f) * sizeof(double complex));

    generate_dummy_data(A, y, N_s, N_f);

    printf("Profiling %s with N_f=%d\n", solver, N_f);

    // Выполнение солвера в цикле
    for (int i = 0; i < iterations; i++) {
        if (strcmp(solver, "gauss_naive") == 0) 
            solve_gauss_naive(N_s, N_f, b_s, A, y, reg, w, Ah, Ah_A, Ah_y, M, BT);
        else if (strcmp(solver, "cholesky_naive") == 0)
            solve_cholesky_naive(N_s, N_f, b_s, A, y, reg, w, Ah, Ah_A, Ah_y, BT);
        else if (strcmp(solver, "gauss_perminov") == 0)
            solve_gauss_perminov(N_s, N_f, b_s, A, y, reg, w, Ah, Ah_A, Ah_y, M, tA, BT, tBT);
        else if (strcmp(solver, "cholesky_perminov") == 0)
            solve_cholesky_perminov(N_s, N_f, b_s, A, y, reg, w, Ah, Ah_A, Ah_y, tA, BT, tBT);
        else if (strcmp(solver, "gauss_winograd") == 0)
            solve_gauss_winograd(N_s, N_f, b_s, A, y, reg, w, Ah, Ah_A, Ah_y, M, rf, cf, BT);
        else if (strcmp(solver, "cholesky_winograd") == 0)
            solve_cholesky_winograd(N_s, N_f, b_s, A, y, reg, w, Ah, Ah_A, Ah_y, rf, cf, BT);
        else if (strcmp(solver, "qr_householder") == 0)
            solve_qr_householder(N_s, N_f, b_s, A, y, reg, w, R, y_ext, v);
    }

    free(A); free(y); free(w); free(Ah); free(Ah_A); free(Ah_y); free(M);
    free(rf); free(cf); free(BT); free(tA); free(tBT); free(R); free(y_ext); free(v);
    return 0;
}