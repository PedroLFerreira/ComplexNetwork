from network import Network
import matplotlib.pyplot as plt
import numpy as np
net = Network()

net.ER_Random(1000,0.2)
net.ShowNodes()

for n in range(net.NodeCount()):
    print(str(n)+str((net.InDegree(n),net.OutDegree(n))))

print(net.ConditionalDegreeDistribution())


#x = np.array([0,2,4,4,5,6])
#y = np.array([1,2,1,7,8,5])
#plt.hist2d(x,y)
plt.show()
