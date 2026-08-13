NVCC   ?= nvcc
# sm_75 = Turing (first IMMA int8 tensor cores); sm_80/86 = Ampere; sm_90 = Hopper.
# Override for your card, e.g. make ARCH=sm_90
ARCH   ?= sm_80
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
