import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time


class Network:
    """ A network of nodes. """
    
    def __init__(self):
        self.nodes = defaultdict(set)

    def Load(self, filename, length = None):
        """ Imports a list of edges to construct network. """
        file = open(filename, 'r')
        self.nodes = defaultdict(set)
        listOfEdges = []
        if length != None:
            listOfEdges = [None]*length
            i=0
            for line in file:
                line = line.split()
                line = [int(x) for x in line]
                listOfEdges[i] = line
                i+=1
        else:
            for line in file:
                line = line.split()
                line = [int(x) for x in line]
                listOfEdges.append(line)
        self.Init(listOfEdges)

    def Save(self, filename):
        """ Saves network in edge list format. """
        file = open(filename,'w')
        print(self.nodes)
        for n in self.nodes:
            print (n)
            for v in self.nodes[n]:
                print(' '+str(v))
                file.write(str(n)+" "+str(v)+"\n")
                

    def Init(self, listOfEdges ):
        """ Initializes graph from edge list. """
        for edge in listOfEdges:
            self.nodes[edge[0]].add(edge[1])
            if edge[1] not in self.nodes:
                self.nodes[edge[1]] = set()


    def ER_Random(self, V, p, undirected=False):
        """ Generates a ER random network with V vertices and probability p of edge occurrence. """
        t = time.clock()
        print("Creating a ER random network with V={} nodes and edge probability p={}.".format(V, p))
        for n in range(V):
            self.nodes[n] = set()
            if((n+1)%(V/100)==0):
                print("      Node Progress: {:.1f}%".format((n+1)/V*100))
        
        if undirected:
            for n in range(V):
                for v in range(n,V):
                    if(v != n and random.uniform(0,1) < p):
                        self.AddEdge(v, n)
                        self.AddEdge(n, v)
                if((n+1)%(V/100)==0):
                    print("      Edge Progress: {:.1f}%\r".format((n+1)/V*100))
        else:
            for n in range(V):
                for v in range(V):
                    if(v != n and random.uniform(0,1) < p):
                        self.AddEdge(v, n)
                if((n+1)%(V/100)==0):
                    print("      Edge Progress: {:.1f}%\r".format((n+1)/V*100))
        t = time.clock() - t
        print("Random Network with {} nodes created in {:.3f} seconds.".format(self.NodeCount(),t))
                    

    def AddEdge(self, fromNode, toNode):
        """ Adds an edge between fromNode to toNode. """
        self.nodes[fromNode].add(toNode)


    def ShowNodes(self):
        """ Prints adjacency list to the console """
        print("\n=======NODES=======")
        for node in self.nodes:
            print(str(node) + ":" + str(self.nodes[node]))
        print("===================\n")

        
    def Degree(self, node):
        """ Computes the degree of the node. """
        if node in self.nodes:
            return len(self.nodes[node])
        return None

    def OutDegree(self, node):
        """ Computes the out-degree of the node. """
        if node in self.nodes:
            return len(self.nodes[node])
        return None
    
    def InDegree(self, node):
        """ Computes the in-degree of the node. """
        outdeg = 0
        for n in self.nodes:
            if node in self.nodes[n]:
                outdeg += 1
        return outdeg

    def AvDegree(self):
        return 2*self.EdgeCount()/self.NodeCount()

    def DegreeDistribution(self):
        """ Computes the degree distribution of the network and draws the plot. """
        maxDegree = 0
        for node in self.nodes:
            maxDegree = max(self.Degree(node), maxDegree)

        distribution = [0]*(maxDegree+1)

        for node in self.nodes:
            degree = self.Degree(node)
            distribution[degree] += 1
        normalization = sum(distribution)
        for d in range(0, len(distribution)):
            distribution[d] = distribution[d] / normalization
        
        plt.plot(range(0,len(distribution)), distribution)
        plt.xlabel("Degree")
        plt.ylabel("Fraction of Nodes")
        plt.show()

        return distribution
    
    def InDegreeDistribution(self):
        maxDegree = 0
        for node in self.nodes:
            maxDegree = max(self.InDegree(node), maxDegree)

        distribution = [0]*(maxDegree+1)

        for node in self.nodes:
            degree = self.InDegree(node)
            distribution[degree] += 1
        normalization = sum(distribution)
        for d in range(0, len(distribution)):
            distribution[d] = distribution[d] / normalization

        plt.plot(range(0,len(distribution)), distribution)
        plt.xlabel("In-Degree")
        plt.ylabel("Fraction of Nodes")
        plt.show()


        return distribution
    
    def OutDegreeDistribution(self):
        maxDegree = 0
        for node in self.nodes:
            maxDegree = max(self.OutDegree(node), maxDegree)

        distribution = [0]*(maxDegree+1)

        for node in self.nodes:
            degree = self.OutDegree(node)
            distribution[degree] += 1
        normalization = sum(distribution)
        for d in range(0, len(distribution)):
            distribution[d] = distribution[d] / normalization

        plt.plot(range(0,len(distribution)), distribution)
        plt.xlabel("Out-Degree")
        plt.ylabel("Fraction of Nodes")
        plt.show()

        return distribution

    def TotalDegree(self, node):
        return self.OutDegree(node) + self.InDegree(node)

    def TotalDegreeDistribution(self):
        maxDegree = 0
        for node in self.nodes:
            maxDegree = max(self.TotalDegree(node), maxDegree)

        distribution = [0]*(maxDegree+1)

        for node in self.nodes:
            totalDegree = self.TotalDegree(node)
            distribution[totalDegree] += 1
        normalization = 1#sum(distribution)
        for d in range(0, len(distribution)):
            distribution[d] = distribution[d] / normalization

        plt.plot(range(0,len(distribution)), distribution)
        plt.xlabel("Total Degree")
        plt.ylabel("Fraction of Nodes")
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

    def ClusteringCoefficient(self, node):
        """ Calculate the clustering Coefficient of a node"""
        # does this work for directional networks?
        coefficient = 0

        # checks everythin 2x, dunno how to be more efficient with sets
        for neibourgh1 in self.nodes[node]:
            for neibourgh2 in self.nodes[node]:
                if neibourgh2 in self.nodes[neibourgh1]:
                    coefficient += 1
        l = len(self.nodes[node])
        if l > 1:
            coefficient /= (l * (l - 1))
        
        return coefficient

    def AvClusteringCoefficient(self):
        """ Calculate the average clustering Coefficient"""
        # does this work for directional networks?
        coefficients = [self.ClusteringCoefficient(i) for i in range(len(self.nodes))]
        
        return sum(coefficients) / len(self.nodes)

    def Lattice(self, shape, periodic=False):
        """ create a regular lattice network """

        nodeSizes = [1]
        for side in shape:
            nodeSizes.append(side*nodeSizes[-1])

        self.nodes = defaultdict(set)

        for i in range(nodeSizes[-1]):
            for (gap, size) in zip(nodeSizes, shape):
                if (i // gap) % size != size-1:
                    self.nodes[i].add(i+gap)
                    self.nodes[i+gap].add(i)
                elif periodic:
                    self.nodes[i].add(i-gap*(size-1))
                    self.nodes[i-gap*(size-1)].add(i)

    def CircularGraph(self, n, k):
        """ create a circular graph """

        self.nodes = defaultdict(set)
        
        for i in range(n):
            for j in range(1, k):
                if i+j > n-1:
                    self.nodes[i].add(i+j-n)
                    self.nodes[i+j-n].add(i)
                else:
                    self.nodes[i].add(i+j)
                    self.nodes[i+j].add(i)

