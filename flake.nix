{
  description = "TensorFlow + CUDA + CleverHans (pip/uv) on NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          pkgs.python310
          pkgs.uv
          pkgs.git
          pkgs.cacert

          # CUDA runtime pieces TF often needs at runtime on NixOS
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cudnn

          # Common native build deps (safe to include; helps when pip needs to compile something)
          pkgs.pkg-config
          pkgs.stdenv.cc.cc
          pkgs.zlib
          pkgs.glib
        ];

        shellHook = ''
          # --- TLS certs (fixes SSL_CERTIFICATE_VERIFY_FAILED) ---
          export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          export SSL_CERT_FILE=$NIX_SSL_CERT_FILE
          export REQUESTS_CA_BUNDLE=$NIX_SSL_CERT_FILE

          # --- Make NVIDIA driver + CUDA libs visible at runtime ---
          # Driver libs on NixOS:
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
          export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}

          # Optional: reduce TF VRAM grab
          export TF_FORCE_GPU_ALLOW_GROWTH=true

          # --- venv ---
          if [ ! -d .venv ]; then
            uv venv
          fi
          source .venv/bin/activate

          echo "DevShell ready. Python: $(python --version)"
          echo "LD_LIBRARY_PATH includes /run/opengl-driver/lib and CUDA libs."
        '';
      };
    });
}
