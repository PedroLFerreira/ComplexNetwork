from network import Network
from math import *


N=100

Net0 = Network()
Net0.setSeed(53)
Net0.ER_Random(100, 0.1/(N-1))
degs0 = [Net0.Degree(x) for x in Net0.nodes]
maxd0 = max(degs0)



Net1 = Network()
Net1.setSeed(53)
Net1.ER_Random(100, 1/(N-1))
degs1 = [Net1.Degree(x) for x in Net1.nodes]
maxd1 = max(degs1)


Net2 = Network()
Net2.setSeed(54)
Net2.ER_Random(100, ((log(N)/N) - 1/(N-1))/1.9 )
degs2 = [Net2.Degree(x) for x in Net2.nodes]
maxd2 = max(degs2)

Net3 = Network()
Net3.setSeed(53)
Net3.ER_Random(100, (log(N)+1)/N)
degs3 = [Net3.Degree(x) for x in Net3.nodes]
maxd3 = max(degs3)

maxD = max(maxd0, maxd1, maxd2, maxd3)
degs0 = [ d/maxD for d in degs0 ]
degs1 = [ d/maxD for d in degs1 ]
degs2 = [ d/maxD for d in degs2 ]
degs3 = [ d/maxD for d in degs3 ]

#Net0.DrawNetwork(useForce=True, colorFilter = degs0)
#Net1.DrawNetwork(useForce=True, colorFilter = degs1)
Net2.DrawNetwork(useForce=True, colorFilter = degs2)
Net3.DrawNetwork(useForce=True, colorFilter = degs3)
