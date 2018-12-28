echo "############################# initial cleanup ############################# "
# cleanup wrapper
# rm -r build
# rm -r lib

echo "#############################  build starspace ############################# "
# build starspace lib
# cd ..
# make clean
# make -f makefile_py
# cd -

echo "#############################  build wrapper ############################# "
# build wrapper
# mkdir lib
# cp ../libstarspace.a ./lib
# mkdir build
cd build
conan install ..
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cd -

echo "#############################  run test ############################# "
# run test
cp ./build/starwrap.so ./test
cd test
python3 test.py
cd -