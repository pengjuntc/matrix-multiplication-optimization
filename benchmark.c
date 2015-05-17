#include <stdlib.h>		// For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>		// For: perror
#include <string.h>		// For: memset

#include <float.h>		// For: DBL_EPSILON
#include <math.h>		// For: fabs

#include <sys/time.h>		// For struct timeval, gettimeofday
#include <gsl/gsl_cblas.h>
#include <unistd.h>

#include "dgemm.h"

extern char *optarg;
extern int optind;
extern int optopt;
extern int opterr;
extern int optreset;

#define MAX_SPEED 26.4

/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the C
 * interface - hence the reference semantics. */
void reference_dgemm(int N, double ALPHA, double A[N][N], double B[N][N],
		     double C[N][N])
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, ALPHA,
		    (double *)A, N, (double *)B, N, 1.0, (double *)C, N);
}

void usage(char *name)
{
	fprintf(stderr, "usage: %s [-h] [-s size] [-o outfile]\n", name);
}

double wall_time()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return 1. * t.tv_sec + 1.e-6 * t.tv_usec;
}

void die(const char *message)
{
	perror(message);
	exit(EXIT_FAILURE);
}

void fill(double *p, int n)
{
	for (int i = 0; i < n; ++i)
		p[i] = 2 * drand48() - 1;	// Uniformly distributed over [-1, 1]
}

void absolute_value(double *p, int n)
{
	for (int i = 0; i < n; ++i)
		p[i] = fabs(p[i]);
}

int *process_sizes(char *arg, int *nitems, int *max)
{
	int num_items, max_val = -1, i;
	int *items = NULL;
	char *p, *curr;

	if (!arg)
		return NULL;

	/* First, count the number of sizes */
	num_items = 0;
	p = arg;
	while (*p) {
		while (*p && *p != ',')
			p++;
		num_items++;
		if (*p)
			p++;
	}

	/* If no items, bail. */
	if (!num_items)
		goto error;

	/* Now create an array of these items. */
	items = malloc(sizeof(int) * num_items);
	curr = p = arg;
	i = 0;
	while (*curr) {
		int val;
		/* Find the end of this item */
		val = strtol(curr, &curr, 10);
		if (val <= 0)
			goto error;
		if (val > max_val)
			max_val = val;
		items[i++] = val;
		if (*curr)
			curr++;
	}

	if (nitems)
		*nitems = num_items;
	if (max)
		*max = max_val;
	return items;

 error:
	if (items)
		free(items);
	*nitems = 0;
	*max = 0;
	return NULL;
}

int default_sizes[] =
 { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
    319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };

int process_arguments(int argc, char **argv, int **sizes_out, int *nsizes_out,
		      int *nmax_out, FILE ** fout)
{
	int *test_sizes = NULL, nsizes, nmax;
	FILE *outfile = NULL;
	char ch;
	while ((ch = getopt(argc, argv, "hs:o:")) != -1) {
		switch (ch) {
		case 'h':
			usage(argv[0]);
			exit(0);
			break;
		case 's':
			test_sizes = process_sizes(optarg, &nsizes, &nmax);
			if (!test_sizes)
				return -1;
			break;
		case 'o':
			outfile = fopen(optarg, "w");
			if (!outfile)
				return -1;
			break;
		default:
			return -1;
		}
	}
	if (!test_sizes) {
		test_sizes = default_sizes;
		nsizes = sizeof(default_sizes) / sizeof(default_sizes[0]);
		nmax = test_sizes[nsizes - 1];
	}
	if (!outfile)
		outfile = stdout;
	*sizes_out = test_sizes;
	*nsizes_out = nsizes;
	*nmax_out = nmax;
	*fout = outfile;
	return 0;
}

/* The benchmarking program */
int main(int argc, char **argv)
{
	int *test_sizes = NULL, nsizes, nmax;
	FILE *outfile = NULL;
	if (process_arguments
	    (argc, argv, &test_sizes, &nsizes, &nmax, &outfile)) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	fprintf(outfile, "Description:\t%s\n", dgemm_desc);

	/* allocate memory for all problems */
	double *buf = NULL;
	buf = (double *)malloc(3 * nmax * nmax * sizeof(double));
	if (buf == NULL)
		die("failed to allocate largest problem size");

	double Mflops_s[nsizes], per[nsizes], aveper, grade;

	/* For each test size */
	for (int isize = 0; isize < nsizes; ++isize) {
		/* Create and fill 3 random matrices A,B,C */
		int n = test_sizes[isize];

		double *A = buf + 0;
		double *B = A + nmax * nmax;
		double *C = B + nmax * nmax;

		fill(A, n * n);
		fill(B, n * n);
		fill(C, n * n);

		/* Measure performance (in Gflops/s). */

		/* Time a "sufficiently long" sequence of calls to reduce noise */
		double Gflops_s, seconds = -1.0;
		double timeout = 0.1;	// "sufficiently long" := at least 1/10 second.
		for (int n_iterations = 1; seconds < timeout; n_iterations *= 2) {
			/* Warm-up */
			square_dgemm(n, (double (*)[])A, (double (*)[])B,
				     (double (*)[])C);

			/* Benchmark n_iterations runs of square_dgemm */
			seconds = -wall_time();
			for (int it = 0; it < n_iterations; ++it)
				square_dgemm(n, (double (*)[])A,
					     (double (*)[])B, (double (*)[])C);
			seconds += wall_time();

			/*  compute Gflop/s rate */
			Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
		}

		/* Storing Mflop rate and calculating percentage of peak */
		Mflops_s[isize] = Gflops_s * 1000;
		per[isize] = Gflops_s*100/MAX_SPEED;

		fprintf(outfile, "Size: %d\tMflop/s: %8g\tPercentage:%6.2lf\n", n,
			Mflops_s[isize], per[isize]);

		/* Ensure that error does not exceed the theoretical error bound. */

		/* C := A * B, computed with square_dgemm */
		memset(C, 0, n * n * sizeof(double));
		square_dgemm(n, (double (*)[])A, (double (*)[])B,
			     (double (*)[])C);

		/* Do not explicitly check that A and B were unmodified on 
		 * square_dgemm exit
		 *  - if they were, the following will most likely detect it:   
		 * C := C - A * B, computed with reference_dgemm 
		 */
		reference_dgemm(n, -1., (double (*)[])A, (double (*)[])B,
				(double (*)[])C);

		/* A := |A|, B := |B|, C := |C| */
		absolute_value(A, n * n);
		absolute_value(B, n * n);
		absolute_value(C, n * n);

		/* C := |C| - 3 * e_mach * n * |A| * |B|, 
		 * computed with reference_dgemm */
		reference_dgemm(n, -3. * DBL_EPSILON * n, (double (*)[])A,
				(double (*)[])B, (double (*)[])C);

		/* If any element in C is positive, then something went wrong 
		 * in square_dgemm */
		for (int i = 0; i < n * n; ++i)
			if (C[i] > 0)
				die("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n");
	}

	free(buf);

	/* Calculating average percentage of peak reached by algorithm */
	aveper=0;
	for (int i=0; i<nsizes;i++)
		aveper+= per[i];
	aveper/=nsizes*1.0;
  
	/* Assigning grade based on average percentage reached (40% gets 75; 
	 * 60% gets 100; rest distributed proportionally) */
	if (aveper >= 60) grade = 100.0;
	else if (aveper >= 40) grade = (aveper-40)*0.25*100.0/20.0 + 75.0;
	else grade = aveper * 2 * 0.75;

	/* Printing average percentage and grade to screen */
	fprintf(outfile, "Average percent peak = %g\tGrade = %g\n", aveper, grade);

	return 0;
}
