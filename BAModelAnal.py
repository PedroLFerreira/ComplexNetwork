from network import Network
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import math
import random

random.seed(9)
x = np.linspace(0,10,100)
var = 0.1
y = [math.sin(i+random.uniform(-0.5,0.5))   for i in x]

print(x)
print(y)
powerLaw = lambda k, g, a: a*k**g
popt, pcov = curve_fit(powerLaw, x, y)
plt.scatter(x,y)
plt.plot(x,np.sin(x))
plt.plot(x,powerLaw(x,*popt),'r--')
plt.show()



net = Network()

net.BA_Random(N=5000)
ddist = net.DegreeDistribution(loglogscale = False)

#def powerLaw(k, gamma):
#    return k**(-gamma)

x = np.linspace(0, len(ddist), num=len(ddist))
x = x[1:]
print(ddist)
y = np.array(ddist[1:])

select = (y != 0)

print(x)
print(len(x))
print(y)
print(len(y))
popt, pcov = curve_fit(powerLaw, x[select], y[select], p0=(-3,1), sigma=np.sqrt(y[select]*5000)/5000)
print(popt)
plt.xscale('log')
plt.yscale('log')
plt.scatter(x,y)
plt.plot(x,powerLaw(x,*popt),'r--')
plt.show()