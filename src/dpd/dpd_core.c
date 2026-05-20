#include 
#include 
#include 

#define A_IDX(row, col, num_cols) ((row) * (num_cols) + (col))
#define R_IDX(row, col, N) ((row) * (N) + (col))

// 1. Генерация матрицы GMP
void create_gmp_matrix_c(const double complex* x, double complex* A, 
                         int n_samples, int p_order, int m_depth, int cross_depth) {
    int col = 0;
    int num_cols = ((p_order + 1) / 2) * m_depth * (2 * cross_depth + 1);
    
    for (int p = 1; p <= p_order; p += 2) {
        for (int m = 0; m < m_depth; m++) {
            for (int l = -cross_depth; l <= cross_depth; l++) {
                for (int n = 0; n < n_samples; n++) {
                    int idx_m = n - m;
                    int idx_l = n - m - l;
                    
                    if (idx_m >= 0 && idx_l >= 0 && idx_l < n_samples) {
                        double magnitude = cabs(x[idx_l]);
                        double power_term = pow(magnitude, p - 1);
                        A[A_IDX(n, col, num_cols)] = x[idx_m] * power_term;
                    } else {
                        A[A_IDX(n, col, num_cols)] = 0.0 + 0.0 * I;
                    }
                }
                col++;
            }
        }
    }
}

// 2. Решатель QR Хаусхолдера
void qr_householder_solver_c(const double complex* A, const double complex* y, double complex* w, 
                             int M, int N, double reg) {
    int total_rows = M + N;
    
    double complex* R = (double complex*)malloc(total_rows * N * sizeof(double complex));
    double complex* y_ext = (double complex*)calloc(total_rows, sizeof(double complex));
    double complex* v = (double complex*)malloc(total_rows * sizeof(double complex));

    if (!R || !y_ext || !v) return;

    double lambda_sqrt = sqrt(reg);
    for (int i = 0; i < total_rows; i++) {
        if (i < M) {
            y_ext[i] = y[i];
            for (int j = 0; j < N; j++) R[R_IDX(i, j, N)] = A[i * N + j];
        } else {
            y_ext[i] = 0.0;
            for (int j = 0; j < N; j++) R[R_IDX(i, j, N)] = (j == (i - M)) ? lambda_sqrt : 0.0;
        }
    }

    for (int i = 0; i < N; i++) {
        double col_norm_sq = 0.0;
        for (int j = i; j < total_rows; j++) {
            double magnitude = cabs(R[R_IDX(j, i, N)]);
            col_norm_sq += magnitude * magnitude;
        }
        
        double col_norm = sqrt(col_norm_sq);
        if (col_norm < 1e-18) continue;

        double complex alpha = R[R_IDX(i, i, N)];
        double complex phase = (cabs(alpha) > 1e-18) ? (alpha / cabs(alpha)) : 1.0;
        double complex sigma = phase * col_norm;

        double v_norm_sq = 0.0;
        for (int j = i; j < total_rows; j++) {
            v[j - i] = (j > i) ? R[R_IDX(j, i, N)] : (alpha + sigma);
            double v_mag = cabs(v[j - i]);
            v_norm_sq += v_mag * v_mag;
        }

        if (v_norm_sq < 1e-18) continue;

        for (int k = i; k < N; k++) {
            double complex dot_vr = 0.0;
            for (int j = i; j < total_rows; j++) {
                dot_vr += conj(v[j - i]) * R[R_IDX(j, k, N)];
            }
            double complex gamma = (2.0 * dot_vr) / v_norm_sq;
            for (int j = i; j < total_rows; j++) {
                R[R_IDX(j, k, N)] -= gamma * v[j - i];
            }
        }

        double complex dot_vy = 0.0;
        for (int j = i; j < total_rows; j++) {
            dot_vy += conj(v[j - i]) * y_ext[j];
        }
        double complex gamma_y = (2.0 * dot_vy) / v_norm_sq;
        for (int j = i; j < total_rows; j++) {
            y_ext[j] -= gamma_y * v[j - i];
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        if (cabs(R[R_IDX(i, i, N)]) > 1e-18) {
            double complex sum = 0.0;
            for (int j = i + 1; j < N; j++) sum += R[R_IDX(i, j, N)] * w[j];
            w[i] = (y_ext[i] - sum) / R[R_IDX(i, i, N)];
        } else {
            w[i] = 0.0;
        }
    }

    free(R); free(y_ext); free(v);
}