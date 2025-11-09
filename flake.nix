{
  description = "A flake.nix to allow easy development with the Aegis-Reborn for Nix afficionados.";
  inputs = {
    cargo2nix.url = "github:cargo2nix/cargo2nix/release-0.12";
    flake-utils.follows = "cargo2nix/flake-utils";
    nixpkgs.follows = "cargo2nix/nixpkgs";
  };

  outputs = inputs: with inputs;
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [cargo2nix.overlays.default];
        };

        rustPkgs = pkgs.rustBuilder.makePackageSet {
          rustVersion = "1.85.0";
          packageFun = import ./Cargo.nix;
        };

      in rec {
        # This section defines the "finished product"
        packages = {
          aegis-reborn = (rustPkgs.workspace.aegis-reborn {});
          default = packages.aegis-reborn;
        };

        # This section defines the "developer workshop"
        devShells.default = pkgs.mkShell {
          # Pulls in all build dependencies (rustc, cargo)
          # from 'aegis-reborn' package.
          inputsFrom = [ packages.aegis-reborn ];

          nativeBuildInputs = [
            pkgs.rust-analyzer
            pkgs.cargo-edit
            pkgs.rustfmt
          ];
        };
      }
    );
}