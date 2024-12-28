set -e
rm -rf build
mkdir build
clang -Wall -Wextra -O2 -g -o build/main tf.c
cp -r assets build
./build/main
