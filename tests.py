from network import Network
import network as nt

nt.np.random.seed(47)
nt.random.seed(47)

net = Network()

#net.CircularGraph(19, 1)
net.ER_Random(V=20, p=0.1, isDirected=True)
#net.Init([[1,0], [2,0], [3,0], [4,0], [5,0], [0,1], [0,2], [0,3], [0,4], [0,5]])
net.ShowNodes()

print(net.AvClusteringCoefficient())

print(net.ClosenessCentrality(0))
print(net.HarmonicCentrality(0))
bd1 = net.BetweennessCentrality()
bd2 = net.BetweennessCentralitySlow()
EC  = net.EigenvectorCentrality()
KATZ= net.KatzCentrality()
PGRK= net.PageRank(alpha=10)

print("compare algorthms:")
print("\t BC1        BC2        EC        KATZ      PGRK")
for key in net.nodes:
    print(key, "{:8f}   {:8f}   {:8f}   {:8f}   {:8f}".format(bd1[key], bd2[key], EC[key], KATZ[key], PGRK[key]), sep='\t')

#print(net.EigenvectorCentrality())

# path = net.ShortestPath(0, 5)
# print(path)

print('average lengh: ', net.AveragePathLenght())
net.DrawNetwork()