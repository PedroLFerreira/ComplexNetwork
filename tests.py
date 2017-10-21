from network import Network

net = Network()

net.CircularGraph(20, 3)
net.ShowNodes()
print(net.AvClusteringCoefficient())

print(net.ClosenessCentrality(0))