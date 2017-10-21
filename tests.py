from network import Network

net = Network()

#net.CircularGraph(20, 1)
net.ER_Random(V=20, p=0.1)
net.ShowNodes()
print(net.Neiborghs(0))
print(net.AvClusteringCoefficient())

print(net.ClosenessCentrality(0))
print(net.HarmonicCentrality(0))
print(net.BetweennessCentrality()[0])