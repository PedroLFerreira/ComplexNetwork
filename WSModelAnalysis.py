from network import Network
import numpy as np

net = Network()

x = np.linspace(0, 1, 50)
avgPathLength = []
for b in x:
    net.WS_Random(N=100, K=1, beta=b)
    avgPathLength.append(net.AveragePathLenght())
    # net.DegreeDistribution()

print(avgPathLength)