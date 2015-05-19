# matrix-multiplication-optimization
Optimize matrix multiplication with several tricks.

command to test the program: ./run.sh

>1. First, determine initial good block sizes for BI, BJ, and BK; the included paper (goto_dgemm.pdf) includes some general advice on this. You might also find it helpful to write a shell script to try out a range of different sizes for BI, BJ, and BK. The supplied block sizes are *not* optimal.


>2. Break dgebb() to work in small subblocks that can be packed into registers efficiently. In particular, break dgebb() work into two subroutines:

 > A. General code that (slowly) computes an arbitrary subblock of C from panels of A and B using simple nested loops.
  
 > B. Code that updates fixed RI by RJ subblocks of C from panels of the current blocks of A and B. Start with simple code to test correctness of subblocking, but then focus on optimizing this code. Your optimized subblock code should start by loading RI x RJ siubblock of C into some small number local variables (e.g. alocal double array), and then loop through all of RI x BK and BK x RJ pieces of A and B, using them to update the local subblock of C. Finally, it should store the updated subblock of C back to main memory. Note that picking RI and RJ determines how many local variables you need for holding pieces of C, B, and A. Choose these numbers carefully so that you do not use more than 16 local floating point variables!

  
>3. Change the the subblock code to use AVX vector registers to compute subblocks instead of scalar registers. This will increase the size of your subblocks, because now each local variables (of type __m256d instead of double), will hold four doubles.  These types and the routines which use them are declared in the header files <emmintrin.h> and  <immintrin.> In particular, start by paying attention to the functions _mm256_loadu_pd(), _mm256_set1_pd(), _mm256_add_pd(), _mm256_mul_pd(), and _mm256_storeu_pd(). Figuring out how to use these options on the 16 vector floating point registers you have available is the key to good performance. Note that you may need to change RI and RJ from step (2) above.


>4. Convert depb() to load the current block of B being worked on into a local BK x BJ buffer that is 32-byte aligned and passed to calls to depbb(). Also change your vector subblock code to use aligned loads (_mm256_load_pd()) when accessing B instead of unaligned loads. To declare an aligned array, use a declaration like this:
static double blocal[BK][BJ]  __attribute__ ((aligned (32)));;
