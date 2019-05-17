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
predictTags
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

### How to predict tags?
If you train a model in trainMode=0, you can get tag predictions. Here is the Python code to get prediction for `barack obama` in the political_social_media dataset.

```
sp.initFromSavedModel('tagged_model')
sp.initFromTsv('tagged_model.tsv')

dict_obj = sp.predictTags('barack obama', 10)
dict_obj = sorted( dict_obj.items(), key = itemgetter(1), reverse = True )

for tag, prob in dict_obj:
    print( tag, prob )
```

And you get this:
```
__label__obama 0.5291043519973755
__label__stopobamasamnesty 0.5073596239089966
__label__florida 0.5003609657287598
__label__entitlement 0.47724902629852295
__label__savesarah 0.4583539664745331
__label__keystone 0.4561977982521057
__label__1866 0.43746984004974365
__label__waroncoal 0.4283019006252289
__label__notleading 0.4117577075958252
__label__cuba 0.3887191414833069
```
For the full example, please refer to `test_predictTags.py` in `test` directory.
