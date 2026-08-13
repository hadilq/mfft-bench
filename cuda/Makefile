NVCC   ?= nvcc
# "native" needs CUDA 11.5+ and targets the installed GPU exactly.
# Override if cross-compiling, e.g. make ARCH=sm_90 (sm_75+ has int8 tensor cores).
ARCH   ?= native
NVFLAGS ?= -O3 -arch=$(ARCH) -lineinfo -Xcompiler -O3

.PHONY: all clean run check

all: gemm_bench

gemm_bench: gemm_bench.cu
	$(NVCC) $(NVFLAGS) -o $@ $< -lcublas

check: gemm_bench
	./gemm_bench --n 512 --reps 2 --check

run: gemm_bench
	./gemm_bench --n 4096 --reps 5

clean:
	rm -f gemm_bench
