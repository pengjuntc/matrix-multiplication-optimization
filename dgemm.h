#ifndef _SQUARE_DGEMM_H
#define _SQUARE_DGEMM_H
extern void square_dgemm(int N, double A[N][N], double B[N][N], double C[N][N]);
extern const char *dgemm_desc;

#endif /* _SQUARE_DGEMM_H */
