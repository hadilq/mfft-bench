CC      ?= cc
CFLAGS  ?= -O3 -march=native -funroll-loops -Wall -Wextra -std=gnu11
LDFLAGS ?=
LDLIBS  ?= -lm
PREFIX  ?= /usr/local
LIMB_BITS ?= 16

CFLAGS += -DLIMB_BITS=$(LIMB_BITS)

SRC := src/kernel.c src/bigmat.c src/roots.c src/methods.c src/mfft.c src/mlgemm.c src/main.c
OBJ := $(SRC:.c=.o)
BIN := mfft-bench

ifdef WITH_OPENMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

ifdef WITH_BLAS
CFLAGS += -DHAVE_CBLAS
LDLIBS += -lopenblas
endif

.PHONY: all clean check install bench

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c src/mfftbench.h
	$(CC) $(CFLAGS) -c -o $@ $<

check: $(BIN)
	./$(BIN) --test-roots
	./$(BIN) --n 8  --bits 128
	./$(BIN) --n 16 --bits 256
	./$(BIN) --n 8  --bits 512 --sigma 3
	./$(BIN) --n 8  --bits 1024
	./$(BIN) --ml --n 128

bench: $(BIN)
	./$(BIN) --n 64 --bits 2048 --no-naive --reps 3

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)$(PREFIX)/bin/$(BIN)

clean:
	rm -f $(OBJ) $(BIN)
