 #include <stdio.h>
 #include <stdlib.h>
 #include <nmmintrin.h>
 #include <string.h>

void sgemm(int m, int n, float *A, float *C) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i += 40) {
            __m128 c[10];
            for (int r = 0; r < 10; r++)
                c[r] = _mm_setzero_ps();
            float *p = A+i;
            for (int k = 0; k < m; k++) {
                __m128 b = _mm_load1_ps(A+j*(n+1)+k*(n));
                for (int r = 0; r < 10; r++) {
                    c[r] = _mm_add_ps(c[r], _mm_mul_ps(_mm_loadu_ps(p + n*k + r*4), b));
                }
            }

            for (int r = 0; r < 14; r++) {
                _mm_storeu_ps(C+i+r*4+j*n, c[r]);
            }
        }
    }

}
