import starwrap as sw
import numpy as np

arg = sw.args()
arg.trainFile = './input.txt'
arg.testFile = './input.txt'
arg.trainMode = 5				

sp = sw.starSpace(arg)
sp.init()
sp.train()
# sp.evaluate()

sp.nearestNeighbor('some text', 10)


sp.saveModel('model')
sp.saveModelTsv('model.tsv')

sp.initFromSavedModel('model')
sp.initFromTsv('model.tsv')

print(np.array(sp.getDocVector('this\tis\ttest', '\t')))
print(np.array(sp.getDocVector('this is test', ' ')))