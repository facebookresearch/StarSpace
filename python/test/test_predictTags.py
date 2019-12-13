import starwrap as sw
import numpy as np
from operator import itemgetter

arg = sw.args()
arg.trainFile = './tagged_post.txt'
arg.trainMode = 0		

sp = sw.starSpace(arg)
sp.init()
sp.train()

sp.nearestNeighbor('barack', 10)


sp.saveModel('tagged_model')
sp.saveModelTsv('tagged_model.tsv')


sp.initFromSavedModel('tagged_model')
sp.initFromTsv('tagged_model.tsv')

dict_obj = sp.predictTags('barack obama', 10)
dict_obj = sorted( dict_obj.items(), key = itemgetter(1), reverse = True )

for tag, prob in dict_obj:
    print( tag, prob )
