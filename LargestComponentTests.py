from network import Network
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.lines as mlines


Net = Network()
Net.setSeed(50)

N=10000
x = np.linspace(0,0.01,100)
y = []
for p in x:
    Net.ER_Random(N,p)
    totalNodes = list(Net.nodes.keys())
    unsearched = list(Net.nodes.keys())
    #print("totalNodes="+str(totalNodes))
    grouped = list()
    connectedComponents = []
    for n in totalNodes:
        if n in unsearched:
            ccc = list(Net.ConnectedComponent(n))
            
            connectedComponents.append(ccc)
            if len(ccc)>N/2:
                break
            grouped.extend(ccc)
            unsearched = [x for x in totalNodes if x not in grouped]

    #print(connectedComponents)
    largestCC = connectedComponents[0]
    for cc in connectedComponents:
        if len(cc) > len(largestCC):
            largestCC = cc
    #print("largestCC="+str(largestCC))
    y.append(len(largestCC)/N)

x = [i*(N-1) for i in x]
print("x="+str(x))
print(len(x))
print("y="+str(y))
print(len(y))
ax = plt.figure()

fill([0,1,1,0], [-0.1,-0.1,1.1,1.1], c='red', alpha=1)
plt.text(s='Subcritical Regime', x=0.5,y=0.6, rotation = 90)
fill([1,math.log(N),math.log(N),1],[-0.1,-0.1,1.1,1.1],c='orange', alpha=1)
plt.text(s='Supercritical Regime', x=4,y=0.6, rotation = 90)
fill([10, math.log(N),math.log(N),10],[-0.1,-0.1,1.1,1.1],'g', alpha=1)
plt.text(s='Connected Regime', x=8,y=0.6, rotation = 90)


plt.plot(x,y,'-o',c='black')
plt.xlabel(r'$<k>=p(N-1)$')
plt.ylabel('Size of the Giant Connected Component / N')
plt.xticks(np.arange(0,11,1))
plt.xlim(0, 10)
plt.ylim(-0.1, 1.1)
plt.show()

#Net.DrawNetwork(useForce=True,forceIterations=10,drawNodeNames=True)