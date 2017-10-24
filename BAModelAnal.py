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

<<<<<<< HEAD
net.BA_Random(N=20)

net.DrawNetwork(useForce = True)
=======
net.BA_Random(N=1000)
ddist = net.DegreeDistribution(loglogscale = False)

#def powerLaw(k, gamma):
#    return k**(-gamma)



x = np.linspace(0, len(ddist), num=len(ddist))
x = x[1:]
print(ddist)
y = ddist[1:]
>>>>>>> bd2e10cb979cfc84f784fcce4b2a37c0e7bfdeef

print(x)
print(len(x))
print(y)
print(len(y))
popt, pcov = curve_fit(powerLaw, x, y,p0=(3,1))
print(popt)
plt.xscale('log')
plt.yscale('log')
plt.scatter(x,y)
#plt.plot(x,powerLaw(x,*popt),'r--')
plt.show()