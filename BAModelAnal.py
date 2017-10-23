from network import Network

net = Network()

net.BA_Random(N=1000)

net.DegreeDistribution(loglogscale = True, cum=True)
