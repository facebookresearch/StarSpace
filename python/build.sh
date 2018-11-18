# cleanup wrapper
rm -r build
rm -r lib

# build starspace lib
cd ..
make clean
make
cd -

# build wrapper
mkdir lib
cp ../libstarspace.so ./lib
mkdir build
cd build
conan install ..
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cd -
