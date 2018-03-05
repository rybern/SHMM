nix-shell -p armadillo eigen3_3 gcc --command "
g++ -c lib/libshmm.cpp
ar rvs libshmm.a libshmm.o
\nm libshmm.a | grep shmm
#rm libshmm.o
mv libshmm.a lib/"
