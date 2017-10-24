from network import Network

net = Network()

net.BA_Random(N=20)

net.DrawNetwork(useForce = True)

net.DegreeDistribution(loglogscale = True, cum=True)
