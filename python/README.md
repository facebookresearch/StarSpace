# Starwrap

This is the python wrapper for Starspace. Which is not complete. Below are the APIs those are done and to be done.
#### done:
```
init
initFromTsv
initFromSavedModel
train
evaluate
getDocVector
nearestNeighbor
saveModel
saveModelTsv
loadBaseDocs
```
#### to be done:
```
getNgramVector
printDoc
predictOne
```

## How to build?
- make sure you have CMake installed. Otherwise install it from [here](https://cmake.org/install/)
- install Conan c++ package manager from [here](https://conan.io/downloads.html).
- now, clone this repository, move to the directory `StarSpace > python`.
- run the build script. 

```
chmod +x build.sh
./build.sh
```

- build script will download necessory packages, build wrapper and run test code. 
- when it is done, you will find `starwrap.so` inside newly created `build` directory. 
- you can either copy `starwrap.so` into your python project or set `LD_LIBRARY_PATH` (GNU/Linux) `DYLD_LIBRARY_PATH` (Mac OSX) and import is as a python module `import starwrap`.

## How to use?
API is very easy and straightforward. Please refer `test.py` in `test` directory.