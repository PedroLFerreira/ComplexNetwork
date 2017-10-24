from network import Network

net = Network()

<<<<<<< HEAD
net.BA_Random(10000, 3)
#net.ShowNodes()
net.DegreeDistribution(loglogscale = True)

#net.DrawNetwork(useForce=False, drawNodeNames=True)
=======
net.BA_Random(N=1000)

net.DegreeDistribution(loglogscale = True, cum=True)
>>>>>>> fde463c89ea07730fcb9000d7c08eaa88a4f8e77
