from network import Network
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

net = Network()
net.ER_Random(100,.1)
#net.Load("undirected.txt")
net.DrawNetwork()

