 #include <stdio.h>
 #include <stdlib.h>
 #include <nmmintrin.h>
 #include <string.h>

void sgemm(int m, int n, int d, float *A, float *C) {
    int i, j = 0;
    int n_tmp = 0;
    int n_40 = n-n%40;
    n_tmp = n-n_40;
    int n_36 = n-n_tmp%36;
    n_tmp = n-n_36;
    int n_20 = n-n_tmp%20;
    n_tmp = n-n_20;
    int n_12 = n-n_tmp%12;
    n_tmp = n-n_12;
    int n_4  = n-n_tmp%4;
    for (j = 0; j < n; j++) {
        for (i = 0; i < n_40; i += 40) {
            __m128 c[10];
            for (int r = 0; r < 10; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 10; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 10; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }

        for (i = n_40; i < n_36; i += 36) {
            __m128 c[9];
            for (int r = 0; r < 9; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 9; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 9; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }

        for (i = n_36; i < n_20; i += 20) {
            __m128 c[5];
            for (int r = 0; r < 5; r+=5) {
                c[r] = _mm_setzero_ps();
                c[r+1] = _mm_setzero_ps();
                c[r+2] = _mm_setzero_ps();
                c[r+3] = _mm_setzero_ps();
                c[r+4] = _mm_setzero_ps();
            }
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 5; r+=5) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                    c[r+1] = _mm_add_ps(c[r+1], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+(r+1)*4), b));
                    c[r+2] = _mm_add_ps(c[r+2], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+(r+2)*4), b));
                    c[r+3] = _mm_add_ps(c[r+3], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+(r+3)*4), b));
                    c[r+4] = _mm_add_ps(c[r+4], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+(r+4)*4), b));
                }
            }

            for (int r = 0; r < 5; r+=5) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
                _mm_storeu_ps(C+i+(r+1)*4+j*n, c[r+1]);
                _mm_storeu_ps(C+i+(r+2)*4+j*n, c[r+2]);
                _mm_storeu_ps(C+i+(r+3)*4+j*n, c[r+3]);
                _mm_storeu_ps(C+i+(r+4)*4+j*n, c[r+4]);
            }
        }

        for (i = n_20; i < n_12; i += 12) {
            __m128 c[3];
            for (int r = 0; r < 3; r+=3) {
                c[r] = _mm_setzero_ps();
                c[r+1] = _mm_setzero_ps();
                c[r+2] = _mm_setzero_ps();
            }

            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 3; r+=3) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                    c[r+1] = _mm_add_ps(c[r+1], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+(r+1)*4), b));
                    c[r+2] = _mm_add_ps(c[r+2], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+(r+2)*4), b));
                }
            }

            for (int r = 0; r < 3; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }

        for (i = n_12; i < n_4; i += 4) {
            __m128 c[1];
            for (int r = 0; r < 1; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 1; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 1; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }
        
        /*
        for (i = n_20; i < n_16; i += 16) {
            __m128 c[2];
            for (int r = 0; r < 2; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 2; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 2; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }   
        }
        
        for (i = n-n%8; i < n-n%4; i += 4) {
            __m128 c[1];
            for (int r = 0; r < 1; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 1; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 1; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }   
        }
        */
        for (i = n_4; i < n; i++) {
            for (int k = 0; k < m; k++) {
                C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
            }
        }
    }

}
