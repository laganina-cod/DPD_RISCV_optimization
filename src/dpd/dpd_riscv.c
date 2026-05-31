#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IDX(row, col, cols) ((row) * (cols) + (col))

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// =====================================================================
// ЯДРА УМНОЖЕНИЯ
// =====================================================================

void matmul_naive(int n, int m, int p, double complex *A, double complex *B, double complex *C, double complex *B_T) {
    #pragma GCC ivdep
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < p; j++) B_T[IDX(j, k, m)] = B[IDX(k, j, p)];
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double complex sum = 0;
            #pragma GCC ivdep
            #pragma GCC unroll 4
            for (int k = 0; k < m; k++) sum += A[IDX(i, k, m)] * B_T[IDX(j, k, m)];
            C[IDX(i, j, p)] = sum;
        }
    }
}

void matmul_perminov(int n, int k_dim, int m, double complex *A, double complex *B, double complex *C, double *tA, double complex *B_T, double *tB_T) {
    #pragma GCC ivdep
    for (int i = 0; i < n * k_dim; i++) tA[i] = creal(A[i]) + cimag(A[i]);
    
    for (int l = 0; l < k_dim; l++) {
        #pragma GCC ivdep
        for (int j = 0; j < m; j++) {
            double complex b_val = B[IDX(l, j, m)];
            B_T[IDX(j, l, k_dim)] = b_val;
            tB_T[IDX(j, l, k_dim)] = creal(b_val) + cimag(b_val);
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double m1 = 0, m2 = 0, m3 = 0;
            #pragma GCC ivdep
            #pragma GCC unroll 4
            for (int l = 0; l < k_dim; l++) {
                m1 += creal(A[IDX(i, l, k_dim)]) * creal(B_T[IDX(j, l, k_dim)]);
                m2 += cimag(A[IDX(i, l, k_dim)]) * cimag(B_T[IDX(j, l, k_dim)]);
                m3 += tA[IDX(i, l, k_dim)] * tB_T[IDX(j, l, k_dim)];
            }
            C[IDX(i, j, m)] = (m1 - m2) + I * (m3 - m1 - m2);
        }
    }
}

void matmul_winograd(int n, int m_dim, int p, double complex *A, double complex *B, double complex *C, double complex *row_f, double complex *col_f, double complex *B_T) {
    int d = m_dim / 2;
    #pragma GCC ivdep
    for (int k = 0; k < m_dim; k++) {
        for (int j = 0; j < p; j++) B_T[IDX(j, k, m_dim)] = B[IDX(k, j, p)];
    }
    
    for (int i = 0; i < n; i++) {
        double complex sum = 0;
        #pragma GCC ivdep
        #pragma GCC unroll 4
        for (int k = 0; k < d; k++) sum += A[IDX(i, 2*k, m_dim)] * A[IDX(i, 2*k+1, m_dim)];
        row_f[i] = sum;
    }
    
    for (int j = 0; j < p; j++) {
        double complex sum = 0;
        #pragma GCC ivdep
        #pragma GCC unroll 4
        for (int k = 0; k < d; k++) sum += B_T[IDX(j, 2*k, m_dim)] * B_T[IDX(j, 2*k+1, m_dim)];
        col_f[j] = sum;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double complex val = 0;
            #pragma GCC ivdep
            #pragma GCC unroll 4
            for (int k = 0; k < d; k++) {
                val += (A[IDX(i, 2*k, m_dim)] + B_T[IDX(j, 2*k+1, m_dim)]) * (A[IDX(i, 2*k+1, m_dim)] + B_T[IDX(j, 2*k, m_dim)]);
            }
            C[IDX(i, j, p)] = val - row_f[i] - col_f[j];
        }
    }
    
    if (m_dim % 2 != 0) {
        for (int i = 0; i < n; i++) {
            #pragma GCC ivdep
            for (int j = 0; j < p; j++) C[IDX(i, j, p)] += A[IDX(i, m_dim-1, m_dim)] * B_T[IDX(j, m_dim-1, m_dim)];
        }
    }
}

// =====================================================================
// РЕШАТЕЛИ (Gauss, Cholesky)
// =====================================================================

void gauss_solve_inplace(int n, double complex *Ah_A, double complex *Ah_y, double complex *w, double complex *M) {
    int cols = n + 1;
    for (int i = 0; i < n; i++) {
        #pragma GCC ivdep
        for (int j = 0; j < n; j++) M[IDX(i, j, cols)] = Ah_A[IDX(i, j, n)];
        M[IDX(i, n, cols)] = Ah_y[i];
    }
    
    for (int i = 0; i < n; i++) {
        int max_r = i; double max_e = cabs(M[IDX(i, i, cols)]);
        for (int k = i + 1; k < n; k++) {
            if (cabs(M[IDX(k, i, cols)]) > max_e) { 
                max_e = cabs(M[IDX(k, i, cols)]); 
                max_r = k; 
            }
        }
        
        if (max_r != i) {
            #pragma GCC ivdep
            for (int j = i; j <= n; j++) { 
                double complex t = M[IDX(i, j, cols)]; 
                M[IDX(i, j, cols)] = M[max_r*cols+j]; 
                M[max_r*cols+j] = t; 
            }
        }
        
        for (int k = i + 1; k < n; k++) {
            double complex c = -M[IDX(k, i, cols)] / M[IDX(i, i, cols)];
            #pragma GCC ivdep
            #pragma GCC unroll 4
            for (int j = i; j <= n; j++) M[IDX(k, j, cols)] += c * M[IDX(i, j, cols)];
        }
    }
    
    for (int i = n - 1; i >= 0; i--) {
        double complex s = 0;
        #pragma GCC ivdep
        for (int j = i + 1; j < n; j++) s += M[IDX(i, j, cols)] * w[j];
        w[i] = (M[IDX(i, n, cols)] - s) / M[IDX(i, i, cols)];
    }
}

void cholesky_solve_inplace(int n, double complex *Ah_A, double complex *Ah_y, double complex *w) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double complex s = 0;
            #pragma GCC ivdep
            #pragma GCC unroll 4
            for (int k = 0; k < j; k++) s += Ah_A[IDX(i, k, n)] * conj(Ah_A[IDX(j, k, n)]);
            
            if (i == j) Ah_A[IDX(i, i, n)] = csqrt(Ah_A[IDX(i, i, n)] - s);
            else Ah_A[IDX(i, j, n)] = (Ah_A[IDX(i, j, n)] - s) / Ah_A[IDX(j, j, n)];
        }
    }
    
    for (int i = 0; i < n; i++) {
        double complex s = 0; 
        #pragma GCC ivdep
        for (int j = 0; j < i; j++) s += Ah_A[IDX(i, j, n)] * w[j];
        w[i] = (Ah_y[i] - s) / Ah_A[IDX(i, i, n)];
    }
    
    for (int i = n - 1; i >= 0; i--) {
        double complex s = 0; 
        #pragma GCC ivdep
        for (int j = i + 1; j < n; j++) s += conj(Ah_A[IDX(j, i, n)]) * w[j];
        w[i] = (w[i] - s) / conj(Ah_A[IDX(i, i, n)]);
    }
}

// =====================================================================
// EXPORT API (7 РЕШАТЕЛЕЙ)
// =====================================================================

#define PREP \
    for(int i=0; i<n_s; i++) { \
        _Pragma("GCC ivdep") \
        for(int j=0; j<n_f; j++) Ah[IDX(j, i, n_s)] = conj(A[IDX(i, j, n_f)]); \
    } \
    memset(Ah_y, 0, n_f * sizeof(double complex)); \
    for(int i=0; i<n_f; i++) { \
        _Pragma("GCC ivdep") \
        _Pragma("GCC unroll 4") \
        for(int j=0; j<n_s; j++) Ah_y[i] += Ah[IDX(i, j, n_s)] * y[j]; \
    }

EXPORT void solve_gauss_naive(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *M, double complex *BT) {
    PREP 
    matmul_naive(n_f, n_s, n_f, Ah, A, Ah_A, BT); 
    for(int i=0; i<n_f; i++) Ah_A[IDX(i, i, n_f)] += reg; 
    gauss_solve_inplace(n_f, Ah_A, Ah_y, w, M);
}

EXPORT void solve_gauss_perminov(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *M, double *tA, double complex *BT, double *tBT) {
    PREP 
    matmul_perminov(n_f, n_s, n_f, Ah, A, Ah_A, tA, BT, tBT); 
    for(int i=0; i<n_f; i++) Ah_A[IDX(i, i, n_f)] += reg; 
    gauss_solve_inplace(n_f, Ah_A, Ah_y, w, M);
}

EXPORT void solve_gauss_winograd(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *M, double complex *rf, double complex *cf, double complex *BT) {
    PREP 
    matmul_winograd(n_f, n_s, n_f, Ah, A, Ah_A, rf, cf, BT); 
    for(int i=0; i<n_f; i++) Ah_A[IDX(i, i, n_f)] += reg; 
    gauss_solve_inplace(n_f, Ah_A, Ah_y, w, M);
}

EXPORT void solve_cholesky_naive(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *BT) {
    PREP 
    matmul_naive(n_f, n_s, n_f, Ah, A, Ah_A, BT); 
    for(int i=0; i<n_f; i++) Ah_A[IDX(i, i, n_f)] += reg; 
    cholesky_solve_inplace(n_f, Ah_A, Ah_y, w);
}

EXPORT void solve_cholesky_perminov(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double *tA, double complex *BT, double *tBT) {
    PREP 
    matmul_perminov(n_f, n_s, n_f, Ah, A, Ah_A, tA, BT, tBT); 
    for(int i=0; i<n_f; i++) Ah_A[IDX(i, i, n_f)] += reg; 
    cholesky_solve_inplace(n_f, Ah_A, Ah_y, w);
}

EXPORT void solve_cholesky_winograd(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *Ah, double complex *Ah_A, double complex *Ah_y, double complex *rf, double complex *cf, double complex *BT) {
    PREP 
    matmul_winograd(n_f, n_s, n_f, Ah, A, Ah_A, rf, cf, BT); 
    for(int i=0; i<n_f; i++) Ah_A[IDX(i, i, n_f)] += reg; 
    cholesky_solve_inplace(n_f, Ah_A, Ah_y, w);
}

EXPORT void solve_qr_householder(int n_s, int n_f, double complex *A, double complex *y, double reg, double complex *w, double complex *R, double complex *y_ext, double complex *v) {
    memset(R, 0, (n_s + n_f) * n_f * sizeof(double complex));
    
    for(int i=0; i<n_s; i++) { 
        #pragma GCC ivdep
        for(int j=0; j<n_f; j++) R[IDX(i, j, n_f)] = A[IDX(i, j, n_f)]; 
        y_ext[i] = y[i]; 
    }
    
    for(int i=0; i<n_f; i++) R[IDX(n_s + i, i, n_f)] = sqrt(reg);
    
    for(int i=0; i<n_f; i++) {
        double col_norm_sq = 0; 
        #pragma GCC ivdep
        for(int j=i; j<n_s+n_f; j++) col_norm_sq += cabs(R[IDX(j, i, n_f)]) * cabs(R[IDX(j, i, n_f)]);
        
        double sigma = (cabs(R[IDX(i, i, n_f)]) > 0 ? (cabs(R[IDX(i, i, n_f)]) / R[IDX(i, i, n_f)]) : 1.0) * sqrt(col_norm_sq);
        v[0] = R[IDX(i, i, n_f)] + sigma; 
        
        #pragma GCC ivdep
        for(int j=i+1; j<n_s+n_f; j++) v[j-i] = R[IDX(j, i, n_f)];
        
        double v_norm_sq = 0; 
        #pragma GCC ivdep
        for(int j=0; j<n_s+n_f-i; j++) v_norm_sq += cabs(v[j]) * cabs(v[j]);
        
        if (v_norm_sq > 1e-20) {
            for(int k=i; k<n_f; k++) {
                double complex dot = 0; 
                #pragma GCC ivdep
                #pragma GCC unroll 4
                for(int j=0; j<n_s+n_f-i; j++) dot += conj(v[j]) * R[IDX(i+j, k, n_f)];
                
                #pragma GCC ivdep
                #pragma GCC unroll 4
                for(int j=0; j<n_s+n_f-i; j++) R[IDX(i+j, k, n_f)] -= (2.0 * dot / v_norm_sq) * v[j];
            }
            double complex dy = 0; 
            #pragma GCC ivdep
            for(int j=0; j<n_s+n_f-i; j++) dy += conj(v[j]) * y_ext[i+j];
            
            #pragma GCC ivdep
            for(int j=0; j<n_s+n_f-i; j++) y_ext[i+j] -= (2.0 * dy / v_norm_sq) * v[j];
        }
    }
    
    for(int i=n_f-1; i>=0; i--) {
        double complex s = 0; 
        #pragma GCC ivdep
        for(int j=i+1; j<n_f; j++) s += R[IDX(i, j, n_f)] * w[j];
        w[i] = (y_ext[i] - s) / R[IDX(i, i, n_f)];
    }
}