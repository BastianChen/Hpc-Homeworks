# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = mpiicc
#OPT = -no-multibyte-chars
OPT = -O3 -ipo -xHost -xAVX2 -qopt-prefetch -qopt-report-phase=vec
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT) -qopenmp -no-multibyte-chars
LDFLAGS = -Wall -qopenmp
# mkl is needed for blas implementation
LDLIBS = $(LDFLAGS) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas benchmark-omp
objects = benchmark-test.o benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o benchmark-omp.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-test : benchmark-test.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-naive : benchmark.o dgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-omp : benchmark.o dgemm-omp.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
