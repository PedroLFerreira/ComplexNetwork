from network import Network
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import math
import random

random.seed(9)

powerLaw = lambda k, g, a: a*k**-(g-1)

def distribution(n, k):
    return 2*k*(k+1) / (n*(n+1)*(n+2))

net = Network()

m=2
n=5000

CC = []
CCerror = []
distributions = []
x = np.linspace(0, 1, 2)
for alpha in x:
    #temp = []
    #for iteration in range(5):
    net.ModifiedBA_Random22(n, m, alpha=alpha)
    ddist = net.DegreeDistribution(showPlot=False, cum=True)
    #temp.append(net.AvClusteringCoefficient())
    distributions.append(ddist)
    #CC.append(sum(temp)/len(temp))
    #CCerror.append(np.std(temp)/len(temp))

degrees = np.arange(0, max(map(len, distributions)), dtype=float)
#plt.errorbar(x, CC, yerr=CCerror)
#plt.show()

plt.xscale('log')
plt.yscale('log')

plt1 = plt.plot(degrees[m:len(distributions[0])], distributions[0][m:], '-o', label=r'$\alpha = 0$')
plt2 = plt.plot(degrees[m:len(distributions[1])], distributions[1][m:], '-o', label=r'$\alpha = 1$')

plt.xlabel('k')
plt.ylabel('comulative probability')
plt.legend()
plt.show()
plt.savefig('distributionModified.pdf')

# c = 2

# n=100
# net.ModifiedBA_Random22(N=n, k=c, alpha=0.5)#, initialNetwork = [[0,1], [1,2], [2,0]])
# ddist = net.DegreeDistribution(loglogscale = False, cum = True, showPlot = False)

# x = np.arange(0, len(ddist), dtype=float)
# x = x[c:]
# y = np.array(ddist[c:])

# popt, pcov = curve_fit(powerLaw, x, y, p0=(3,1))
# print(popt)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.xscale('log')
# plt.yscale('log')
# plt.scatter(x,y)
# plt.plot(x,powerLaw(x,*popt),'r--')
# plt.ylabel('comulative probability')
# plt.xlabel('k')

# ax.text(50, .1, "$\\gamma={:.3f}$\n$a={:.3f}$".format(popt[0], popt[1]), fontsize=15)

# plt.savefig('BApowerlog.pdf')
# plt.show()

# print(net.AvClusteringCoefficient())

# net.DrawNetwork(useForce=True)

# b = net.BetweennessCentrality()
# d = dict(((key, net.Degree(key)) for key in net.nodes))

# bv, dv = [], []
# for key in net.nodes:
#     bv.append(b[key])
#     dv.append(d[key])

# plt.scatter(dv, bv)
# plt.ylabel('betweenness centrality')
# plt.xlabel('degree')
# plt.xscale('log')
# plt.show()
