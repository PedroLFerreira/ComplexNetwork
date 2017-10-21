from network import Network

net = Network()

#net.CircularGraph(20, 1)
net.ER_Random(V=20, p=0.1)
#net.Init([[1,0], [2,0], [3,0], [4,0], [5,0], [0,1], [0,2], [0,3], [0,4], [0,5]])
net.ShowNodes()
net.DrawNetwork()
print(net.Neiborghs(0))
print(net.AvClusteringCoefficient())

print(net.ClosenessCentrality(0))
print(net.HarmonicCentrality(0))
print(net.BetweennessCentrality())
print(net.EigenvectorCentrality())