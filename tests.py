from network import Network

net = Network()

net.CircularGraph(20, 1)
net.ShowNodes()
print(net.AvClusteringCoefficient())

print(net.ClosenessCentrality(0))
print(net.HarmonicCentrality(0))