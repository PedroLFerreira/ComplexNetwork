from network import Network
import matplotlib.pyplot as plt
import numpy as np
net = Network()

net.ER_Random(100,0.2)
net.ShowNodes()

for n in range(net.NodeCount()):
    print(str(n)+str((net.InDegree(n),net.OutDegree(n))))

print(net.ConditionalDegreeDistribution())

