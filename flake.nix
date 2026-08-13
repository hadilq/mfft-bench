{
  description =
    "Benchmark of MFFT (Matrix Fast Fourier Transform) against other matrix multiplication methods";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;      # required for the CUDA toolkit
          config.cudaSupport = false;     # only the dev shell needs it
        };
        inherit (pkgs) lib stdenv;

        # -march=native is great for a local benchmark and terrible for a
        # reproducible build product, so it is only the default in the dev
        # shell; the packaged build sticks to a portable baseline.
        mkBench = { limbBits ? 16, withBlas ? true, native ? false }:
          stdenv.mkDerivation {
            pname = "mfft-bench${lib.optionalString (limbBits != 16)
              "-limb${toString limbBits}"}";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [ pkgs.gnumake ];
            buildInputs = lib.optional withBlas pkgs.openblas;

            makeFlags = [
              "CC=${stdenv.cc.targetPrefix}cc"
              "LIMB_BITS=${toString limbBits}"
              "PREFIX=${placeholder "out"}"
              "WITH_OPENMP=1"
            ] ++ lib.optional withBlas "WITH_BLAS=1";

            CFLAGS = "-O3 -funroll-loops -Wall -Wextra -std=gnu11"
              + lib.optionalString native " -march=native";

            enableParallelBuilding = true;

            doCheck = true;
            checkTarget = "check";

            meta = with lib; {
              description =
                "Exact big-integer matrix multiplication benchmark: MFFT vs schoolbook, limb-plane and Strassen";
              homepage = "https://hadilq.com/posts/matrix-fast-fourier-transform/";
              license = licenses.mit;
              platforms = platforms.unix;
              mainProgram = "mfft-bench";
            };
          };

        bench = mkBench { };
        benchLimb1 = mkBench { limbBits = 1; };
      in {
        packages = {
          default = bench;
          mfft-bench = bench;
          # LIMB_BITS=1 reproduces the post's literal base-2 "digit" model,
          # so its m = 16 example can be measured directly.
          mfft-bench-bits = benchLimb1;
        };

        apps.default = {
          type = "app";
          program = "${bench}/bin/mfft-bench";
        };

        checks.default = bench;

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            gcc
            clang
            gnumake
            openblas
            pkg-config
            gdb
            valgrind
            hyperfine
            python3
          ] ++ lib.optionals (stdenv.isLinux && stdenv.isx86_64) [
            # GPU half of the benchmark: make cuda && ./cuda/gemm_bench
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.libcublas
          ] ++ lib.optionals stdenv.isLinux [
            linuxPackages.perf
          ];

          # so `make WITH_BLAS=1` finds cblas.h / libopenblas without fuss
          shellHook = ''
            export CFLAGS="-O3 -march=native -funroll-loops -Wall -Wextra -std=gnu11"
            export LDFLAGS="-L${pkgs.openblas}/lib"
            export C_INCLUDE_PATH="${pkgs.openblas.dev or pkgs.openblas}/include:$C_INCLUDE_PATH"
            echo "mfft-bench dev shell"
            echo "  make && ./mfft-bench --test-roots"
            echo "  make WITH_BLAS=1 && ./mfft-bench --n 64 --bits 8192 --no-naive --no-verify"
            echo "  make cuda && ./cuda/gemm_bench --n 4096 --check"
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
          '';
        };

        formatter = pkgs.nixfmt-classic;
      });
}
