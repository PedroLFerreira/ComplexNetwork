from network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
from collections import defaultdict

net = Network()
net.ER_Random(50,0.04)
#net.Load("undirected.txt")

#ax = plt.axes()
#ax.arrow(0.2, 0.4, 0.1, 0.1, head_width=0.05, head_length=0.1, fc='k', ec='k')

#plt.show()



net.DrawNetwork()

