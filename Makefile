shmm: src/shmm.cpp
	nix-shell -p armadillo eigen3_3 gcc --command "g++ $$NIX_CFLAGS_COMPILE src/shmm.cpp -o shmm"
