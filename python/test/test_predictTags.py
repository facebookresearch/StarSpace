import starwrap as sw
import numpy as np

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

print( sp.predictTags('barack', 4) )