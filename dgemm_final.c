//Written by Jun Peng. 02/19/2015.

#include "dgemm.h"
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>

/* Cache blocking parameters */
#ifndef BI
#define BI 32
#endif

#ifndef BJ
#define BJ 32
#endif 

#ifndef BK
#define BK 96
#endif

#ifndef RI
#define RI 2
#endif

#ifndef RJ
#define RJ 16
#endif

#ifndef RJJ
#define RJJ RJ/4
#endif


#define xstr(s) str(s)
#define str(s) #s

#define CBLOCK_DESCR BI * BJ * BK
#define RBLOCK_DESCR RI * RJ

const char * dgemm_desc = "DGEMM with cache block:" xstr(CBLOCK_DESCR) " register-block:" xstr(RBLOCK_DESCR) " and vectorization";

inline int min(int i, int j)
{
	return i < j ? i : j;
}

/*****
 * General, mildly optimized versions of dgebb and dgepb
 * that can be used in the general case. An optimized implementation
 * would register block and vectorize dgebb for specific block sizes.
 */

/* Multiply a BI*BK block of A times a BK*BJ block of B 
 * into an BI * BJ block of C. In the common case, BI = BJ = BK */

void dgebb_subblock_opt(int bi, int bj, int bk,
                        int Astride, double A[][Astride],  
                        int Bstride, double B[][Bstride], 
                        int Cstride, double C[][Cstride])
{

        int i, j, k;
    
        __m256d a_vect, b_vect[RJJ], c_vect[RI][RJJ];

        for (i = 0; i < RI; i++)
                for(j = 0; j < RJJ; j++)
                        c_vect[i][j] = _mm256_loadu_pd(&C[i][4*j]);

        for(k = 0; k < bk; k++){
                for (j = 0; j < RJJ; j++) {
                        b_vect[j] = _mm256_load_pd(&B[k][4*j]);
                }
                for(i = 0; i < RI; i++){
                        //a_vect = _mm256_broadcast_sd(&A[i][k]);
                        a_vect = _mm256_set1_pd(A[i][k]);
                        for(j = 0; j < RJJ; j++) {
                                c_vect[i][j] = _mm256_add_pd(_mm256_mul_pd(a_vect, b_vect[j]), c_vect[i][j]);
                        }
                }
        }

        for (i = 0; i < RI; i++)
                for (j = 0; j < RJJ; j++) {
                        _mm256_storeu_pd(&C[i][4*j], c_vect[i][j]);
                }
}



void dgebb_subblock_gen(int bi, int bj, int bk,
	   int Astride, double A[][Astride],  
	   int Bstride, double B[][Bstride], 
	   int Cstride, double C[][Cstride])
{
        int i, j, k;
        double tmp;
                
        for(k = 0; k < bk; k++) {              
                for (i = 0; i < bi; i++) {
                        tmp = A[i][k];
                        for(j = 0; j < bj; j++){
                                C[i][j] += tmp*B[k][j];
                        }       
                }
        }
}


void dgebb(int bi, int bj, int bk,
	   int Astride, double A[][Astride],  
	   int Bstride, double B[][Bstride], 
	   int Cstride, double C[][Cstride])
{

        int i, j;
        //int k;
	for (i = 0; i < bi; i+= RI) {
                int ri = min(RI, bi - i); 
                if (ri == RI) {
                        for (j = 0; j < bj; j+=RJ) {
                                int rj = min(RJ, bj - j);
                                if (rj == RJ) {
                                        dgebb_subblock_opt(RI, RJ, bk,
                                                           Astride, (double (*)[])&A[i][0],
                                                           Bstride, (double (*)[])&B[0][j],
                                                           Cstride, (double (*)[])&C[i][j]);
                                } else {
                                        dgebb_subblock_gen(ri, rj, bk,
                                                           Astride, (double (*)[])&A[i][0],
                                                           Bstride, (double (*)[])&B[0][j],
                                                           Cstride, (double (*)[])&C[i][j]);
 
                                }   
                        }
                }
                else {
                        dgebb_subblock_gen(ri, bj, bk,
                                           Astride, (double (*)[])&A[i][0],
                                           Bstride, B,
                                           Cstride, (double (*)[])&C[i][0]);
                }
        }
}



/* Multiply a NxBK panel of A times a BKxBJ block of B
 * into a NxBJ panel of C. */
void dgepb(int N, int bj, int bk, 
	   int Astride, double A[N][Astride],  
	   int Bstride, double B[N][Bstride], 
	   int Cstride, double C[N][Cstride])
{
	int i, j, k;
        static double blocal[BK][BJ] __attribute__ ((aligned (32)));
        for (k = 0; k < BK; k++){
                for (j= 0; j < BJ; j++) {
                        blocal[k][j] = B[k][j];
                }
        }

	for (i = 0; i < N; i += BI) {
		int bi = min(BI, N - i);
		dgebb(bi, bj, bk, 
		      Astride, (double (*)[])&A[i][0],
		      BJ, blocal,
		      Cstride, (double (*)[])&C[i][0]);
	}
}




/* Breaks dgemm down into panels of A and C and blocks of B */

void square_dgemm(int N, double A[N][N], double B[N][N], double C[N][N])
{
	int j, k;
	for (k = 0; k < N; k += BK) {
		int bk = min(BK, N - k);
		for (j = 0; j < N; j += BJ)  {
			int bj = min(BJ, N - j);
			dgepb(N, bj, bk,
			      N, (double (*)[N])&A[0][k], 
			      N, (double (*)[N])&B[k][j],
			      N, (double (*)[N])&C[0][j]);
		}
	}
}



