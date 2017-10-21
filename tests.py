from network import Network

net = Network()


net.CircularGraph(20, 3)
net.ShowNodes()
print(net.AvClusteringCoefficient())

