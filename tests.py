from network import *

net = Network()
net.Load('edgeList.txt')
net.ShowNodes()
net.Save('newEdgeList.txt')