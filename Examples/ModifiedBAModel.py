from network import Network 
import matplotlib.pyplot as plt
import numpy as np

net = Network()
net.setSeed(18)




net.ModifiedBA_Random(100, PAFunction=net.HarmonicCentrality)
net.BetweennessCentrality()
net.DrawNetwork(useForce=True)

net.DegreeDistribution(loglogscale=True)
