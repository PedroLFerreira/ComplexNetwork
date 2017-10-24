from network import Network

net = Network()

net.BA_Random(10000, 3)
#net.ShowNodes()
net.DegreeDistribution(loglogscale = True)

#net.DrawNetwork(useForce=False, drawNodeNames=True)