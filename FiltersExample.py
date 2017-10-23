from network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
from collections import defaultdict


net = Network()
net.WS_Random(100, 1, 0.2)

sfltr = [None]*net.NodeCount()
for node in net.nodes:
    sfltr[node] = net.Degree(node)
sfltr = [sfltr[n]/max(sfltr)*0.01+0.05 for n in net.nodes]

#cfltr = [None]*net.NodeCount()
#for node in net.nodes:
#    cfltr[node] = net.HarmonicCentrality(node)

cfltr = net.BetweennessCentrality()
maxFilter= max(cfltr.values())
cfltr = [cfltr[n]/maxFilter for n in net.nodes]

net.DrawNetwork(useForce=True, drawNodeNames = False, forceIterations = 10, colorFilter = cfltr, sizeFilter = sfltr)

net.DegreeDistribution(showPlot = False, loglogscale = True)



