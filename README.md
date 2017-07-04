# StarSpace

StarSpace is a library for efficient learning of entity representations from relations among collections of discrete entities. 

In the general case, it embeds objects of different types into a vectorial embedding space,
hence the star ('*') and space in the name, and in that space compares them against each other.
It learns to rank a set of entities/documents or objects
given a query entity/document or object, which is not necessarily the same type as the
items in the set.

Use cases:
- Ranking of sets of entities/documents or objects, e.g. ranking web documents.
- Content-based or Collaborative filtering-based Recommendation, e.g. recommending music or videos.
- Text classification, or any other labeling task.
- Embedding graphs, e.g. multi-relational graphs such as Freebase.

# Requirements

StarSpace builds on modern Mac OS and Linux distributions. Since it uses C++11 features, it requires a compiler with good C++11 support. These include :

* (gcc-4.6.3 or newer) or (clang-3.3 or newer)

Compilation is carried out using a Makefile, so you will need to have a working **make**.

You need to install <a href=http://www.boost.org/>Boost</a> library and specify the path of boost library in makefile in order to run StarSpace.

Optional: if one wishes to run the unit tests in src directory, <a href=https://github.com/google/googletest>google test</a> is required and its path needs to be specified in 'TEST_INCLUDES' in makefile.

# Building StarSpace

In order to build StarSpace, use the following:

    git clone https://github.com/facebookresearch/Starspace.git
    cd starspace/src
    make starspace

# File Format

StarSpace takes input files of the following format. 
Each line will be one input example, in the simplest case the input has k words, and each
labels 1..r is a single word:

    word_1 word_2 ... word_k __label__1 ... __label__r

This file format is the same as in <a href="https://github.com/facebookresearch/fastText">fastText</a>. It assumes labels are words that are prefixed by the string \_\_label\_\_, and the prefix string can be set by "-label" argument. 

In order to learn the embeddings, do:

    $./starspace train -trainFile data.txt -model model

where data.txt is a training file containing utf-8 encoded text. At the end of optimization the program will save two files: model and model.tsv. model.tsv is a standard tsv format file containing the entity vectors, one per line. model is a binary file containing the parameters of the model along with the dictionary and all hyper parameters. The binary file can be used later to compute entity vectors or to run evaluation tasks.

In the more general case, each label also consists of words:

    word_1 word_2 ... word_k <tab> label_1_word_1 label_1_word_2 ... <tab> label_r_word_1 .. 

Embedding vectors will be learned for each word and label to group similar inputs and labels together. 

In order to learn the embeddings in the more general case where each label consists words, do:

    $./starspace train -trainFile data.txt -model model -isLabelFeatured true


## Training Mode

StarSpace supports the following two training mode (default is the first one):
* trainMode = 0: Each example contains both input and labels
* trainMode = 1: Each example contains a collection of labels. At training time, one label from the collection is randomly picked as label, and the rest of labels in the collection becomes input.

The use cases of the 2nd train mode will be explained in Example use cases.

# Example use cases

## TagSpace word / tag embeddings

**Setting:** Learning the mapping from a short text to relevant hashtags.

**Model:** the relation goes from bags of words to bags of tags, by learning an embedding of both. 
For instance,  the input “restaurant has great food <\tab> #restaurant <\tab> #yum” will be translated into the following graph. (Nodes in the graph are entities which embeddings will be learned, and edges in the graph are relationships between the entities).

![word-tag](https://github.com/facebookresearch/Starspace/blob/master/examples/tagspace.png)

**Input file format**:

    restaurant has great food #yum #restaurant

**Command:**

    $./starspace train -trainFile input.txt -model tagspace -label '#'


## PageSpace user / page Embeddings 

**Setting:** On Facebook, users can fan public pages they're interested in. We want to embed pages based on their fan data. Having page embeddings can help with page recommendations, for example. 

**Model：** Users are represented as the bag of pages that they fan. Pages are embedded directly.

![user-page](https://github.com/facebookresearch/Starspace/blob/master/examples/user-page.png)

Each user is represented by bag-of-pages fanned by the user.

**Input file format:**

    page_1 page_2 ... page_M

At training time, one random page is selected as a label and the rest of bag of pages are selected as input. This can be achieved by setting flag -trainMode to 1. 

**Command:**

    $./starspace train -trainFile input.txt -model pagespace -label 'page' -trainMode 1


## Document Recommendations

**Setting:** We want to embed and recommend web documents for users based on their historical click data. 

**Model:** Each document is represented by bag of words in the document. Each user is represeted as the document s/he clicked in history. 
At trainint time, one random document is selected as a label and the rest of bag of documents are selected as input. 

![user-doc](https://github.com/facebookresearch/Starspace/blob/master/examples/user-doc.png)


**Input file format:**

    hello world <tab> good morning <tab> good night
    
**Command:**

    ./starspace train -trainFile input.txt -model docspace -trainMode 1 -isLabelFeatured true

# Full Documentation
    
    The following arguments are mandatory for train: 
      -trainFile       training file path
      -model           output model file path

    The following arguments are mandatory for eval: 
      -testFile        test file path
      -model           model file path

    The following arguments for the dictionary are optional:
      -minCount        minimal number of word occurences [1]
      -minCountLabel   minimal number of label occurences [1]
      -ngrams          max length of word ngram [1]
      -bucket          number of buckets [2000000]
      -label           labels prefix [__label__]. See file format section.

    The following arguments for training are optional:
      -trainMode       takes value in [0, 1], see Training Mode Section. [0]
      -lr              learning rate [0.01]
      -dim             size of embedding vectors [10]
      -epoch           number of epochs [5]
      -negiSearchLimit number of negatives sampled [50]
      -maxNegSamples   max number of negatives in a batch update [10]
      -loss            loss function {hinge, softmax} [hinge]
      -margin          margin parameter in hinge loss. It's only effective if hinge loss is used. [0.05]
      -similarity      takes value in [cosine, dot]. Whether to use cosine or dot product as similarity function in  hinge loss.
                       It's only effective if hinge loss is used. [cosine]
      -thread          number of threads [10]
      -adagrad         whether to use adagrad in training [0]
      -isLabelFeatured whether the label contains feature. [0]

    The following arguments are optional:
      -verbose         verbosity level [0]
      -debug           whether it's in debug mode [0]

Note:
We use the same implementation of Ngrams for words as of in <a href="https://github.com/facebookresearch/fastText">fastText</a>.
