CC = gcc 
OPT = -O3 -mavx -mfpmath=sse -funroll-all-loops
CFLAGS += -Wall -std=gnu99 -DGETTIMEOFDAY $(OPT)
LDLIBS = -lgslcblas

targets = benchmark_final
objects = benchmark.o dgemm_final.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

benchmark_final : benchmark.o dgemm_final.o 
	$(CC) $(OPT) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout *~
