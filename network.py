import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time


class Network:
    """ A network of nodes. """
    
    def __init__(self):
        self.nodes = defaultdict(set)

    def Init(self, listOfEdges ):
        """ Initializes graph from edge list. """
        for edge in listOfEdges:
            #if edge[0] not in self.nodes:
            #    self.nodes[edge[0]] = set()
            #if edge[1] not in self.nodes:
            #    self.nodes[edge[1]] = set()
            
            self.nodes[edge[0]].add(edge[1])
            self.nodes[edge[1]].add(edge[0])


    def Random(self, V, p):
        """ Generates a ER random network with V vertices and probability p of edge occurrence. """

        t = time.clock()

        print("Creating a ER random network with V={} nodes and edge probability p={}.".format(V, p))
        for n in range(V):
            #self.nodes[n] = set()
            #print(str(n)+"%"+str(5)+"="+str(n%5))
            if((n+1)%(V/100)==0):
                print("      Node Progress: {:.1f}%".format((n+1)/V*100))
        
        for n in range(V):
            for v in range(n,V):
                if(v != n and random.uniform(0,1) < p):
                    self.AddEdge(v,n)
            if((n+1)%(V/100)==0):
                print("      Edge Progress: {:.1f}%\r".format((n+1)/V*100))
        t = time.clock() - t
        print("Random Network with {} nodes created in {:.3f} seconds.".format(self.NodeCount(),t))
                    

    def AddEdge(self, fromNode, toNode):
        """ Adds an edge between fromNode to toNode. ADD EXCEPTIONS!!!"""
        self.nodes[fromNode].add(toNode)
        self.nodes[toNode].add(fromNode)


    def ShowNodes(self):
        """ Prints adjacency list to the console """
        print("\n=======NODES=======")
        for node in self.nodes:
            print(str(node) + ":" + str(self.nodes[node]))
        print("===================\n")

        
    def Degree(self, node):
        """ Computes the degree of the node """
        if node in self.nodes:
            return len(self.nodes[node])
        return -1

    def AvDegree(self):
        return 2*self.EdgeCount()/self.NodeCount()

    def DegreeDistribution(self):
        """ Computes the degree distribution of the network and draws the plot """
        maxDegree = 0
        for node in self.nodes:
            maxDegree = max(self.Degree(node), maxDegree)
        
        distribution = [0]*(maxDegree+1)
        
        for node in self.nodes:
            degree = self.Degree(node)
            distribution[degree] += 1
        for d in range(0, len(distribution)):
            distribution[d] = distribution[d] / sum(distribution)
        plt.plot(range(0,len(distribution)), distribution)
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()
        return distribution
    
    def EdgeCount(self, verbose = 0):
        """ Returns the number of edges in the network """
        t = time.clock()
        E = 0
        for n in self.nodes:
            
            E += len(self.nodes[n])

            if((n+1)%(len(self.nodes)/50)==0 and verbose == 1):
                print("Counting edges: {:.1f}%\r".format((n+1)/len(self.nodes)*100))
        t = time.clock() - t
        print("EdgeCount took {:.5f} seconds.".format(t))
        return E/2

    def NodeCount(self):
        """ Returns the number of nodes/vertices in the network """
        return len(self.nodes)

    def Density(self):
        """ Computes the density of the network """
        V = self.NodeCount()
        E = self.EdgeCount()
        return 2*(E - V + 1)/(V*(V - 3) + 2)
    
    def ConnectedComponent(self, start):
        """ Using BFS algorithm, returns set of nodes  """
        visited, queue = set(), [start]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.nodes[node] - visited)
        return visited
    
    def ShortestPath(self, start, goal):
        """ Returns shortest path from start to goal"""
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if self.nodes[node]-set(path) != set():
                for next in self.nodes[node]-set(path):
                    if next == set():
                        continue
                    if next == goal:
                        return path + [next]
                    else:
                        queue.append((next, path + [next]))
            else:
                break
    
    def Diameter(self):
        """ Calculate diameter using BFS search. O(n^2) """
        N = len(self.nodes)
        diameterPath = []
        
        for n in range(N):
            for v in range(n, N):
                short = self.ShortestPath(n, v)
                if short != None and len(short) > len(diameterPath):
                    diameterPath = short
        return (len(diameterPath), diameterPath)

    def Lattice(self, shape, periodic=False):
        """ periodic not working """

        nodeSizes = [1]
        for side in shape:
            nodeSizes.append(side*nodeSizes[-1])
        
        print(list(zip(nodeSizes, shape)))

        for i in range(nodeSizes[-1]):
            for (gap, size) in zip(nodeSizes, shape):
                if (i // gap) % size != size-1:
                    self.nodes[i].add(i+gap)
                    self.nodes[i+gap].add(i)
                elif periodic:
                    self.nodes[i].add(i-gap*(size-1))
                    self.nodes[i-gap*(size-1)].add(i)

        # for dim in shape:
        #     for i in range(dim):
        #         if i != dim-1:
        #             self.nodes[i].add(i+1)
        #             self.nodes[i+1].add(i)
        #         elif periodic:
        #             self.nodes[i].add(0)
        #             self.nodes[0].add(i)


#V = 10
#p = 0.1

#net = Network()

#net.Random(V, p)
#net.Init([[0,1],[0,2],[1,3],[4,5]])
#net.ShowNodes()

#degDist = net.DegreeDistribution()
#print("Diameter={}\nLongest Shortest Path:{}".format(net.Diameter()[0],net.Diameter()[1]))
