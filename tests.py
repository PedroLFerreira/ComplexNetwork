from network import Network

net = Network()

net.Random(V = 10, p = .2)

net.ShowNodes()

net = Network()

net.Lattice(shape = [5, 5], periodic=True)

net.ShowNodes()
net.Load('edgeList.txt')
net.ShowNodes()
net.Save('newEdgeList.txt')
