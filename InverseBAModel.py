from network import Network 
import matplotlib.pyplot as plt
import numpy as np

net = Network()
net.setSeed(18)


def invDeg():
    d = net.DegreeDistribution(showPlot = False)
    d = [1/(net.Degree(x)+0.1) for x in net.nodes]
    return d

net.ModifiedBA_Random(10000,1,PAFunction=invDeg)
#net.DrawNetwork(useForce=True)

net.DegreeDistribution(loglogscale=True)
