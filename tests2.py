from network import Network

net = Network()

net.ER_Random(100,0.2)
net.ShowNodes()

for n in range(net.NodeCount()):
    print(str(n)+str((net.InDegree(n),net.OutDegree(n))))

print(net.ConditionalDegreeDistribution())