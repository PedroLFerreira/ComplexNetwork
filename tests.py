from network import Network

net = Network()

net.Load("undirected.txt")
net.ShowNodes()
print(net.Degree(2))

net.Load("directed.txt")
net.ShowNodes()
print((net.InDegree(2),net.OutDegree(2)))

