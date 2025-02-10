cd build || exit
pwd
cmake -DADDITIONAL_COMPILE_OPTIONS="-march=native" -DENABLE_TESTING=OFF -DVTUNE=ON -DTHREAD_COUNT=8 ..
make
