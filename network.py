import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from collections import defaultdict, deque
import random
import time
import numpy as np
import matplotlib.lines as mlines
import math

class Network:
    """ A network of nodes. """
    
    def __init__(self):
        self.nodes = defaultdict(set)
        self.isDirected = False

    def Load(self, filename, length = None, isDirected = False):
        """ Imports a list of edges to construct network. """
        self.isDirected = isDirected
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
                #line = [int(x) for x in line]
                listOfEdges.append(line)
        self.Init(listOfEdges, isDirected)

    def Save(self, filename):
        """ Saves network in edge list format. """
        file = open(filename,'w')
        print(self.nodes)
        for n in self.nodes:
            print (n)
            for v in self.nodes[n]:
                print(' '+str(v))
                file.write(str(n)+" "+str(v)+"\n")
                

    def Init(self, listOfEdges, isDirected = False):
        """ Initializes graph from edge list. """
        self.isDirected = isDirected
        if isDirected:
            for edge in listOfEdges:
                self.nodes[edge[0]].add(edge[1])
                if edge[1] not in self.nodes:
                    self.nodes[edge[1]] = set()
        else:
            print("is undirected ")
            for edge in listOfEdges:
                self.nodes[edge[0]].add(edge[1])
                self.nodes[edge[1]].add(edge[0])
                if edge[1] not in self.nodes:
                    self.nodes[edge[1]] = set()
                if edge[0] not in self.nodes:
                    self.nodes[edge[0]] = set()


    def ER_Random(self, V, p, isDirected=False):
        """ Generates a ER random network with V vertices and probability p of edge occurrence. """
        self.isDirected = isDirected
        t = time.clock()
        print("Creating a ER random network with V={} nodes and edge probability p={}.".format(V, p))
        for n in range(V):
            self.nodes[n] = set()
            if((n+1)%(V/100)==0):
                print("      Node Progress: {:.1f}%".format((n+1)/V*100))
        
        if not self.isDirected:
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
        print("ER-Random Network with {} nodes created in {:.3f} seconds.".format(self.NodeCount(),t))

    def WS_Random(self, N, K, beta):
        """ Generates a WS random network with N, K, beta parameters. """
        self.isDirected = False
        t = time.clock()
        print("Creating a WS random network with N={} nodes, K={} initial neighbors and relink probability beta={}.".format(N, K, beta))
        self.CircularGraph(N, K)
        #print(self.nodes)

        for n in self.nodes:
            buffer = self.nodes[n].copy()
            #print("n={}".format(n))
            for v in buffer:
                #print("   v={}".format(v))
                if random.uniform(0,1) < beta:
                    r = random.choice(list(self.nodes.keys()-buffer-{v,n}))
                    #print("      r={}".format(r))
                    #print("        Before:")
                    #print("          node n:"+str(self.nodes[n]))
                    #print("          node v:"+str(self.nodes[v]))
                    #print("          node r:"+str(self.nodes[r]))
                    #print()
                    self.nodes[n].remove(v)
                    self.nodes[v].remove(n)
                    self.nodes[n].add(r)
                    self.nodes[r].add(n)
                    #print("        After:")
                    #print("          node n:"+str(self.nodes[n]))
                    #print("          node v:"+str(self.nodes[v]))
                    #print("          node r:"+str(self.nodes[r]))
        #print(self.nodes)
        t = time.clock() - t
        print("WS-Random Network with {} nodes created in {:.3f} seconds.\n".format(N, t))

            


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

    def Neighborhood(self, node):
        """ Returns the first neighbors of the node. """
        inNeighbors = set()
        for n in self.nodes:
            if node in self.nodes[n]:
                inNeighbors.add(n)
        
        return inNeighbors | self.nodes[node]

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
    
    def ConditionalDegreeDistribution(self):
        maxInDegree = 0
        maxOutDegree = 0
        for node in self.nodes:
            maxInDegree = max(self.InDegree(node), maxInDegree)
            maxOutDegree = max(self.OutDegree(node), maxOutDegree)
        distribution = np.zeros((maxInDegree + 1,maxOutDegree + 1))

        print(np.shape(distribution))
        for node in self.nodes:
            inDegree = self.InDegree(node)
            outDegree = self.OutDegree(node)
            distribution[inDegree, outDegree] += 1
        normalization = np.sum(distribution)
        for i in range(0, len(distribution)):
            for o in range(0, len(distribution[0])):
                distribution[i, o] = distribution[i, o] / normalization

        plt.xlabel("In-Degree")
        plt.ylabel("Out-Degree")
        plt.imshow(distribution,origin=(0,0))
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
        """ Returns shortest path from start to goal.
            crashes if goal or path not in self.nodes
        """

        visited, previous = {}, {}
        for i in self.nodes:
            visited[i] = False
            previous[i] = None

        Q = deque([start])
        visited[start] = True

        leave = False
        while len(Q) != 0 and not leave:
            v = Q.popleft()

            for neighbor in self.nodes[v]:
                if visited[neighbor] == False:
                    Q.append(neighbor)
                    visited[neighbor] = True
                    previous[neighbor] = v
                    if neighbor == goal:
                        leave = True
                        break

        if previous[goal] == None:
            return None
        
        path = [goal]
        while previous[path[-1]] != None:
            path.append(previous[path[-1]])

        return path[::-1]
    
    def ShortestPaths(self, start):
        """ Distances and shortest paths from start to the other nodes """
        distance = {}
        for i in self.nodes:
            distance[i] = -1

        S = []

        previous = defaultdict(list)
        paths = defaultdict(int)
        paths[start] = 1

        Q = deque([start])
        distance[start] = 0

        while len(Q) != 0:
            v = Q.popleft()
            S.append(v)

            for neighbor in self.nodes[v]:
                if distance[neighbor] == -1:
                    Q.append(neighbor)
                    distance[neighbor] = distance[v] + 1
                if distance[neighbor] == distance[v] + 1:
                    paths[neighbor] += paths[v]
                    previous[neighbor].append(v)

        return distance, previous, paths, S

    def AveragePathLenght(self):
        totalDistance = 0
        for i in self.nodes:
            distance, _, _, _ = self.ShortestPaths(i)
            if -1 in distance.values():
                return float('inf')
            totalDistance += sum(distance.values())

        return totalDistance / (len(self.nodes) * (len(self.nodes) - 1))

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
        for neiborgh1 in self.nodes[node]:
            for neiborgh2 in self.nodes[node]:
                if neiborgh2 in self.nodes[neiborgh1]:
                    coefficient += 1
        l = len(self.nodes[node])
        if l > 1:
            coefficient /= (l * (l - 1))
        
        return coefficient

    def AvClusteringCoefficient(self):
        """ Calculate the average clustering Coefficient"""
        # does this work for directional networks?
        coefficients = [self.ClusteringCoefficient(i) for i in self.nodes]
        
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
        """ create a circular graph 
            k is the number of closest neiborgh, so the <k> of the 
            network generated will be <k>=2k
        """

        self.nodes = defaultdict(set)
        
        for i in range(n):
            for j in range(1, k+1):
                if i+j > n-1:
                    self.nodes[i].add(i+j-n)
                    self.nodes[i+j-n].add(i)
                else:
                    self.nodes[i].add(i+j)
                    self.nodes[i+j].add(i)

    def ClosenessCentrality(self, node):
        """ Calculate the closeness centrality for a given node.
            calculates using the path from node to target
        """
        distance, _, _, _ = self.ShortestPaths(node)

        total = 0
        for other in distance:
            if distance[other] == -1:
                return 0
            else:
                total += distance[other]

        return (len(self.nodes) - 1) / total
    
    def HarmonicCentrality(self, node):
        """ Calculate the harmonic centrality for a given node. """
        distance, _, _, _ = self.ShortestPaths(node)

        total = 0
        for other in distance:
            if distance[other] == -1 or other == node:
                continue
            else: 
                total += 1/distance[other]
        return total / (len(self.nodes) - 1)
    
    def BetweennessCentrality(self):
        """ Calculate the betweenness centrality for all nodes. """
        # i think this is an aproximation. Maybe calculate the proper value?
        CB = defaultdict(int)
        for i in self.nodes:
            distance, previous, paths, S = self.ShortestPaths(i)

            delta = defaultdict(int)

            while len(S) != 0:
                w = S.pop()
                for v in previous[w]:
                    delta[v] += paths[v]/paths[w]*(1 + delta[w])
                if w != i:
                    CB[w] += delta[w]
        
        n = len(self.nodes)
        for key in CB:
            CB[key] *= 1 / ((n - 1) * (n - 2))

        return CB

    def BetweennessCentralitySlow(self):
        """ Calculate the betweenness centrality for all nodes. """
        # i think this is an aproximation. Maybe calculate the proper value?
        CB = defaultdict(int)

        dDistance, dPrevious, dPaths, dS = {},{},{},{}

        for i in self.nodes:
            distance, previous, paths, S = self.ShortestPaths(i)

            dDistance[i] = distance
            dPrevious[i] = previous
            dPaths[i] = paths
            dS[i] = S

        for s in self.nodes:
            for v in self.nodes:
                for t in self.nodes:
                    if not (s != v != t):
                        continue
                    if dDistance[s][t] == dDistance[s][v] + dDistance[v][t]:
                        CB[v] += dPaths[s][v]*dPaths[v][t]/dPaths[s][t]
        
        n = len(self.nodes)
        for key in CB:
            CB[key] *= 1 / ((n - 1) * (n - 2))

        return CB
    
    def EigenvectorCentrality(self, epsilon=1e-6, max_iter=100):
        """ Calculate the Eigenvector centrality for all nodes. """
        M = np.zeros(shape = (len(self.nodes), len(self.nodes)))
        
        traslationIn = {}

        for i,n in enumerate(self.nodes):
            traslationIn[n] = i

        for i in self.nodes:
            for j in self.nodes[i]:
                M[traslationIn[i], traslationIn[j]] = 1

        x = np.ones(shape=len(self.nodes))

        for i in range(max_iter):
            x_i = np.dot(M.T, x)

            x_i /= np.linalg.norm(x_i)

            if (np.sum(abs(x_i-x)) < epsilon):
                break

            x = x_i
        else:
            print('did not converge')

        return dict(zip(self.nodes.keys(),x))

    def KatzCentrality(self, alpha=.9, epsilon=1e-6, max_iter=100):
        """ Calculate the Katz centrality for all nodes """
        M = np.zeros(shape = (len(self.nodes), len(self.nodes)))
        
        traslationIn = {}

        for i,n in enumerate(self.nodes):
            traslationIn[n] = i

        for i in self.nodes:
            for j in self.nodes[i]:
                M[traslationIn[i], traslationIn[j]] = 1

        x = np.ones(shape=len(self.nodes))

        for i in range(max_iter):
            x_i = alpha * np.dot(M.T, x) + np.ones(len(x))

            x_i /= np.linalg.norm(x_i)

            if (np.sum(abs(x_i-x)) < epsilon):
                break

            x = x_i
        else:
            print('did not converge')

        return dict(zip(self.nodes.keys(),x))

    def PageRank(self, alpha=.9, epsilon=1e-6, max_iter=100):
        """ Calculate PageRank centrality for all nodes """
        M = np.zeros(shape = (len(self.nodes), len(self.nodes)))
        
        traslationIn = {}

        for i,n in enumerate(self.nodes):
            traslationIn[n] = i

        for i in self.nodes:
            for j in self.nodes[i]:
                M[traslationIn[i], traslationIn[j]] = 1

        kout = np.array([max(self.OutDegree(i), 1) for i in self.nodes])

        x = np.ones(shape=len(self.nodes))

        for i in range(max_iter):
            x_i = alpha * np.dot(M.T, x/kout) + np.ones(len(x))

            x_i /= np.linalg.norm(x_i)

            if (np.sum(abs(x_i-x)) < epsilon):
                break

            x = x_i
        else:
            print('did not converge')

        return dict(zip(self.nodes.keys(),x))

    #""" DRAWING STUFF """
    def DrawNetwork(self):
        positions = defaultdict(set)
        i = 0
        for node in self.nodes:
            i+=1
            x,y = random.uniform(0,10), random.uniform(0,10)
            positions[node] = [x,y]

        attractionMultiplier = 5
        globalAttractionMultiplier = 0.01
        restLength = 0.5
        globalRestLength = 3
        repulsionMultiplier = .1
        forceMultiplier = 1

        ax = plt.gca()        

        ax.set_facecolor((0.1, 0.1, 0.1))

        print("Starting force-based layout algorithm...")
        t = time.clock()
        iterations = 50
        for i in range(iterations):
            
            print(str(i/iterations*100)+"%", end='\r')
            for node in self.nodes:
                x = positions[node][0]
                y = positions[node][1]
                for n in self.nodes:
                    if n==node:
                        continue
                    u = positions[n][0]
                    v = positions[n][1]
                    d = math.sqrt((x-u)**2+(y-v)**2)
                    
                    repulsion = min(repulsionMultiplier/d**3, 2)

                    xForce = -repulsion

                    if n in self.Neighborhood(node):
                        attraction = max(attractionMultiplier*math.log(d/restLength),0)
                        xForce = attraction-repulsion
                    else:
                        globalAttraction = max(globalAttractionMultiplier*math.log(d/globalRestLength),0)
                        xForce = globalAttraction-repulsion

                    angle = math.atan2(v-y, u-x)
                    x = x + forceMultiplier*xForce*math.cos(angle)/(i+1)
                    y = y + forceMultiplier*xForce*math.sin(angle)/(i+1)
                        
                positions[node] = [x,y]
        print("Force-based layout algorithm finished in "+str(time.clock() - t)+" seconds.")
        
        print("Drawing links...")
        t = time.clock()
        if self.isDirected:
            for n in self.nodes:
                for v in self.nodes[n]:
                    #self._DrawLink(positions[n],positions[v], ax)
                    self._DrawArrow(positions[n], positions[v], ax)
        else:
            for n in self.nodes:
                for v in self.nodes[n]:
                    self._DrawLink(positions[n],positions[v], ax)
                    #self._DrawArrow(positions[n], positions[v], ax)
        print("Links drawn in "+str(time.clock() - t)+" seconds.")

        maxFltr = 1e-6
        fltr = {}
        cmap = plt.get_cmap('summer')
        for node in self.nodes:
            fltr[node] = self.Degree(node)
            #print(fltr[node])
            maxFltr = max(maxFltr, fltr[node])

        t = time.clock()
        print("Drawing nodes...")
        for node in self.nodes:
            ax.add_patch(self._DrawNode(positions[node][0], positions[node][1], radius = .05, color = cmap(fltr[node]/maxFltr)))
            ax.text(positions[node][0], positions[node][1]+0.1, s=str(node), color='gray')
        print("Nodes drawn in "+str(time.clock() - t)+" seconds.")

        #ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        
        plt.axis('scaled')
        plt.show()




    def _DrawNode(self, x, y, radius = 0.1, color = (0.960784, 0.968627, 0.886275)):
        node = plt.Circle((x, y), radius = radius, color = color)
        return node
        
    def _DrawLink(self, p1, p2, ax):
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], alpha = 0.2,zorder = 0)
        ax.add_line(l)
        return l

    def _DrawArrow(self, p1, p2, ax):
        ax.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], head_width = 0.05, head_length = 0.1, color = (1,1,1), alpha = 0.1, length_includes_head = True, zorder = 0)


    






































