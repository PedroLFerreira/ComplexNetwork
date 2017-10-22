from network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
from collections import defaultdict

net = Network()
net.BA_Random(50)

sfltr = [None]*net.NodeCount()
for node in net.nodes:
    sfltr[node] = net.Degree(node)
sfltr = [sfltr[n]/max(sfltr)*0.1+0.05 for n in net.nodes]

cfltr = [None]*net.NodeCount()
for node in net.nodes:
    cfltr[node] = net.HarmonicCentrality(node)
cfltr = [cfltr[n]/max(cfltr) for n in net.nodes]

net.DrawNetwork(useForce=True, forceIterations = 10, colorFilter = cfltr, sizeFilter = sfltr)

net.DegreeDistribution()



