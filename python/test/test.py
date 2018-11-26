import starwrap as sw
arg = sw.args()
arg.trainFile = './input.txt'
arg.trainMode = 5

sp = sw.starSpace(arg)
sp.init()
sp.train()

sp.nearestNeighbor('some text', 10)


sp.saveModel('model')
sp.saveModelTsv('model.tsv')
