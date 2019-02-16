import starwrap as sw
import numpy as np

arg = sw.args()
arg.K=100
sp = sw.starSpace(arg)
# sp.evaluate()




sp.initFromSavedModel('')
#sp.initFromTsv('')

sp.loadBaseDocs()

vec=sp.parseDoc_(""""""))
a= sp.predictOne_(vec)
for i in a:
    par= sp.baseDocs_[i[1]]
    print (i[0],sp.printDoc_(par) )
