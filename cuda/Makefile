NVCC   ?= nvcc

# Architecture selection.
#
# -arch=native asks nvcc to target the installed GPU, which fails outright
# when the GPU is newer than the toolkit (an RTX 50-series card reports
# compute_120, and only CUDA 12.8+ can emit code for it).  Instead: find the
# newest architecture this nvcc actually supports, emit SASS for it, and also
# emit PTX so the driver can JIT forward onto anything newer.  That is how the
# sm_80 build managed to run on sm_120 in the first place.
#
# Override with e.g. make ARCH=sm_90 to pin one architecture.
MAXCC := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed -n 's/^compute_//p' | sort -n | tail -1)

ifneq ($(ARCH),)
GENCODE := -arch=$(ARCH)
else ifneq ($(MAXCC),)
GENCODE := -gencode arch=compute_$(MAXCC),code=sm_$(MAXCC) \
           -gencode arch=compute_$(MAXCC),code=compute_$(MAXCC)
else
GENCODE := -arch=sm_70
endif

NVFLAGS ?= -O3 $(GENCODE) -lineinfo -Xcompiler -O3

.PHONY: all clean run check arch

all: gemm_bench

arch:
	@if [ -n "$(MAXCC)" ]; then 	    echo "nvcc supports up to compute_$(MAXCC)"; 	else 	    echo "nvcc did not report supported architectures; using a safe default"; 	fi
	@echo "flags:$(GENCODE)"

gemm_bench: gemm_bench.cu
	$(NVCC) $(NVFLAGS) -o $@ $< -lcublas

check: gemm_bench
	./gemm_bench --n 512 --reps 2 --check

run: gemm_bench
	./gemm_bench --n 4096 --reps 5

clean:
	rm -f gemm_bench
