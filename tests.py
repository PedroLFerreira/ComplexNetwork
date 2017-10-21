from network import Network

net = Network()

net.Load("undirected.txt")
net.ShowNodes()
print(net.Degree(2))
net.DegreeDistribution()


net.Load("directed.txt")
net.ShowNodes()

for n in range(net.NodeCount()):
    print(str(n)+str((net.InDegree(n),net.OutDegree(n))))

print(net.OutDegreeDistribution())
#print(net.InDegreeDistribution())

print(net.InDegreeDistribution())
print(net.TotalDegreeDistribution())

net.CircularGraph(20, 3)
net.ShowNodes()
print(net.AvClusteringCoefficient())

