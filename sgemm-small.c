 #include <stdio.h>
 #include <stdlib.h>
 #include <nmmintrin.h>
 #include <string.h>

void sgemm(int m, int n, int d, float *A, float *C) {
    int i, j = 0;
    for (j = 0; j < n; j++) {
        for (i = 0; i < n-n%40; i += 40) {
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
        for (i = 0; i < n-n%20; i += 20) {
            __m128 c[5];
            for (int r = 0; r < 5; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 5; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 5; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }
        for (i = 0; i < n-n%16; i += 16) {
            __m128 c[4];
            for (int r = 0; r < 4; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 4; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 4; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }
        for (i = 0; i < n-n%12; i += 12) {
            __m128 c[3];
            for (int r = 0; r < 3; r++)
                c[r] = _mm_setzero_ps();
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 3; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(A+i+n*k+r*4), b));
                }
            }

            for (int r = 0; r < 3; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }   
        }
        for (i = 0; i < n-n%8; i += 8) {
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
        for (i = 0; i < n-n%4; i += 4) {
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

        for (i = n-n%4; i < n; i++) {
            for (int k = 0; k < m; k++) {
                C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
            }
        }
    }

}
