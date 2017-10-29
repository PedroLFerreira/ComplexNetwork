from network import Network
import numpy as np
import matplotlib.pyplot as plt

net = Network()

x = np.logspace(-3, 0, 50)
avgPathLength = []
clusteringCoef = []
for b in x:
    net.WS_Random(N=1000, K=4, beta=b)
    avgPathLength.append(net.AveragePathLenght())
    clusteringCoef.append(net.AvClusteringCoefficient())
    #net.DegreeDistribution()
maxPathLength = max(avgPathLength)
print(maxPathLength)

avgPathLength = [i/maxPathLength for i in avgPathLength]
apl = plt.scatter(x,avgPathLength)
acc = plt.scatter(x,clusteringCoef)
plt.legend((apl, acc), ('<'+r'$\ell$>',r'$<C>$'))
plt.xscale('log')
plt.xlabel(r'$\beta$')
plt.ylabel('Clustering / Path Length')
plt.xlim(1e-3,1)
plt.ylim(0,1)
print(avgPathLength)
print(clusteringCoef)
plt.show()