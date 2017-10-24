from network import Network
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

WSNet = Network()
ERNet = Network()



pA = [1e-4, 1e-3, 1.5e-3,3e-3]
bA = [0.01, 0.1, 0.3, 0.5]
ERDist = []
WSDist = []
"""
for p in pA:
    ERNet.ER_Random(N=10000, p=p)
    ERDist.append(ERNet.DegreeDistribution(showPlot=False))


ax = plt.figure().gca()
ax.annotate(r'$N=10^{4}$',xy=(24,0.3))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ERDistPlot0 = plt.plot(range(0,len(ERDist[0])), ERDist[0], '-o')
ERDistPlot1 = plt.plot(range(0,len(ERDist[1])), ERDist[1], '-o')
ERDistPlot2 = plt.plot(range(0,len(ERDist[2])), ERDist[2], '-o')
ERDistPlot3 = plt.plot(range(0,len(ERDist[3])), ERDist[3], '-o')
plt.legend((r'$p=10^{-4}$',r'$p=10^{-3}$',r'$p=1.5$'+r'$\times$'+r'$10^{-3}$',r'$p=3$'+r'$\times$'+r'$10^{-3}$'))
plt.xlim(1,1e3)
plt.ylim(1e-4,1)
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xlabel("k")
plt.ylabel("P(k)")
plt.show()
"""
for b in bA:
    WSNet.WS_Random(N=10000, K=4, beta=b)
    WSDist.append(WSNet.DegreeDistribution(showPlot=False))

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.annotate(r'$N=10^{4}$',xy=(4,0.6))
WSDistPlot0 = plt.plot(range(0,len(WSDist[0])), WSDist[0], '-o')
WSDistPlot1 = plt.plot(range(0,len(WSDist[1])), WSDist[1], '-o')
WSDistPlot2 = plt.plot(range(0,len(WSDist[2])), WSDist[2], '-o')
WSDistPlot3 = plt.plot(range(0,len(WSDist[3])), WSDist[3], '-o')
plt.legend((r'$\beta=10^{-2}$',r'$\beta=10^{-1}$',r'$\beta=3$'+r'$\times$'+r'$10^{-1}$',r'$\beta=5$'+r'$\times$'+r'$10^{-1}$'))
plt.xlim(1, 1e3)
plt.ylim(1e-4, 1)
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xlabel("k")
plt.ylabel("P(k)")
plt.show()







"""
plt.legend((apl, acc), ('<'+r'$\ell$>',r'$<C>$'))
plt.xscale('log')
plt.xlabel(r'$\beta$')
plt.ylabel('Clustering / Path Length')
plt.xlim(1e-3,1)
plt.ylim(0,1)

plt.show()"""