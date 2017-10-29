from network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
from collections import defaultdict


net = Network()
net.BA_Random(10)

sfltr = [None]*net.NodeCount()
for node in net.nodes:
    sfltr[node] = net.Degree(node)
sfltr = [sfltr[n]/max(sfltr)*0.09+0.01 for n in net.nodes]
cfltr = net.HarmonicCentrality()
maxFilter= max(cfltr.values())
cfltr = [cfltr[n]/maxFilter for n in net.nodes]


net.DegreeDistribution(showPlot = True, loglogscale = False)
net.ShowNodes()
net.DrawNetwork(useForce=True, drawNodeNames = True, forceIterations = 10, colorFilter = cfltr, sizeFilter = sfltr)



