#!/usr/bin/python3

from PyQt5.QtCore import QLineF, QPointF




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import copy


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def gather(self, matrix, cities):

		notInPath = []
		notInPath = [i for i in range(len(cities))]

		toBreak = False
		# find two things that point to each other
		for i in range(len(cities)):
			if toBreak == True:
				break

			for j in range(len(cities)):
				if j == i:
					continue

				if matrix[i,j] != math.inf and matrix[j,i] != math.inf:
					edgePath = [(i,j), (j,i)] # this path is made up of the edges not the nodes
					nodePath = [i,j]
					notInPath.remove(i)
					notInPath.remove(j)
					toBreak = True
					break

		# for each edge find a node that can replace the edge
		e = -1
		while e < len(edgePath) - 1:
			e += 1

			for i in notInPath:

				if matrix[edgePath[e][0], i] != math.inf and matrix[i, edgePath[e][1]] != math.inf:
					edgePath.insert(e, (i, edgePath[e][1]))
					edgePath.insert(e, (edgePath[e+1][0], i))
					del edgePath[e+2]
					nodePath.insert(e+1, i)
					notInPath.remove(i)
					e = -1
					break

			if len(nodePath) == len(cities):
				break


		return nodePath

	
	
	def greedy( self,time_allowance=60.0 ):
		start_time = time.time()
		results = {}
		cities = self._scenario.getCities
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None

		currentSpot = count
		masterMatrix = self.createMatrix(cities)
		path = []
		cityMatrix = np.array(copy.deepcopy(masterMatrix))
		cityMatrix[0,:] = math.inf
	
		for city in cities:
			path.append(currentSpot)#adds first spot

			#what if all are infinity
			min_in_column = np.min(cityMatrix,axis=0) #returns minimum values
			min_index_column= np.argmin(cityMatrix, axis = 0)#returns index of min values
			min_spot = min_index_column[currentSpot]
			cityMatrix[currentSpot][min_spot] = math.inf
			cityMatrix[:,currentSpot] = math.inf
			cityMatrix[min_spot,:] = math.inf
			currentSpot = min_spot # update current spot
		route = []

		for i in range(ncities):
			route.append(cities[path[i]])

		bssf = TSPSolution(route)

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		start_time = time.time()
		results = {}
		default = self.defaultRandomTour()
		#greedy = self.greedy()
		#lowest_cost = greedy['cost']
		num_updates = 0
		max_heap_len = 1
		pruned = 0
		num_sul = 0
		count = 0
		total_created = 0
		bssf = default['cost']#math.inf
		finalnode = node()
		heap = []
		heapq.heapify(heap)
		solution = node()
		solutions = []

		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False

		masterMatrix = self.createMatrix(cities)
		mutationMatrix = copy.deepcopy(masterMatrix)
		currentCities = []
		for x in range(0, len(masterMatrix)):
			currentCities.append(x)

		bound = self.reduceMatrix(mutationMatrix,currentCities)

		currentNode = node()
		currentNode.current_city = 0
		currentNode.bound = bound
		currentNode.path = [0]
		currentNode.RCM = copy.deepcopy(mutationMatrix)
		currentNode.remaining_cities = []

		for x in range(1, len(masterMatrix)):
			currentNode.remaining_cities.append(x)

		heapq.heappush(heap,currentNode)




		while len(heap) and time.time()-start_time < time_allowance:
			currentNode = heapq.heappop(heap)
			#print(currentNode.bound)
			if currentNode.bound > bssf:
				pruned+=1

			else:
				for x in range(0,len(currentNode.remaining_cities)):
					total_created+=1
					newnode = node()
					newnode.remaining_cities = []
					for y in range(0, len(currentNode.remaining_cities)):
						if x != y:
							newnode.remaining_cities.append(currentNode.remaining_cities[y])
						else:
							newnode.path = copy.deepcopy(currentNode.path)
							newnode.path.append(currentNode.remaining_cities[y])
							newnode.current_city = currentNode.remaining_cities[y]
					newnode.bound = currentNode.bound+currentNode.RCM[currentNode.current_city][currentNode.remaining_cities[x]]
					newnode.RCM = self.chooseCity(currentNode.RCM,currentNode.current_city,currentNode.remaining_cities[x])
					if newnode.remaining_cities != 0:

						newnode.bound = newnode.bound+ self.reduceMatrix(newnode.RCM,newnode.remaining_cities)


					if newnode.bound  > bssf:
						pruned+=1
					else:
						if len(newnode.remaining_cities) == 0:
							num_sul+=1
							solution=newnode
							bssf = newnode.bound
							foundTour = True
						else:
							heapq.heappush(heap,newnode)
							if len(heap) > max_heap_len:
								max_heap_len = len(heap)
					count+=1
		end_time = time.time()
		results['cost'] = solution.bound if foundTour else bssf
		results['time'] = end_time - start_time
		results['count'] = num_sul
		if foundTour:
			cityList = [cities[i] for i in solution.path]
			results['soln'] = TSPSolution(cityList)
		else:
			results['soln'] = default['soln']

		results['max'] = max_heap_len
		results['total'] = total_created
		results['pruned'] = pruned
		return results

	def two_opt(self, solution):
        	path = solution.path
        	cost = solution.bound
        	improved = True
        	# while route keeps changing
	        while improved:
        	    improved = False
	            # for all edges
        	    for i in range(1, len(path) - 2):
                	# for all other edges
	                for j in range(i + 1, len(path)):
        	            if j - i == 1:
                	        continue
	                    cities = self._scenario.getCities()
        	            newPath1 = path.copy()
                	    # make new array of cities {1,2,7,5,8}
	                    for k in range(j-i):
        	                newPath1[i+k+1] = path[j-k]
                	    # make other array
	                    newPath2 = path.copy()
        	            for m in range(len(path)):
                	        if m > 0:
                        	    newPath2[m] = newPath1[len(newPath1) - m]
	                    # get cost of visiting cities
        	            path1Cost = self.generateCost(newPath1)
                	    path2Cost = self.generateCost(newPath2)
	                    newCost = min(path1Cost, path2Cost)
        	            newPath = newPath1
                	    if newCost == path2Cost:
                        	newPath = newPath2
	                    #  if new cost is less than old cost
        	            if newCost < cost:
                	        # update path, cost
                        	cost = newCost
                        	path = newPath
                        	improved = True
        	solution.path = path
        	solution.cost = cost

    	def generateCost(self, path):
        	cost = 0
	        cities = self._scenario.getCities()
        	for i in range(len(path)):
	            city = cities[path[i]]
        	    nextCity = cities[path[0]]
	            if i != len(path) - 1:
        	        nextCity = cities[path[i+1]]
	            cost += city.costTo(nextCity)
        	return cost

	def chooseCity(self, matrix, currentrow,destrow):
		newmatrix= copy.deepcopy(matrix)
		for x in range(0,len(matrix)):
			newmatrix[x][destrow] = math.inf
			newmatrix[currentrow][x] = math.inf
		newmatrix[destrow][currentrow] = math.inf
		return newmatrix


	def reduceMatrix(self, matrix, citiesLeft):
		min_row = np.min(matrix, axis =1)
		bound = 0
		for i in range(0,len(citiesLeft)):
			row = citiesLeft[i]
			bound+=min_row[row]
			for j in range(0,len(citiesLeft)):
				col = citiesLeft[j]
				if min_row[row] == math.inf:
					matrix[row][col] = math.inf
				else:
					matrix[row][col] = matrix[row][col] - min_row[row]

		min_col = np.min(matrix, axis=0)

		for i in range(0,len(citiesLeft)):
			row = citiesLeft[i]
			bound += min_col[row]
			for j in range(0,len(citiesLeft)):
				col = citiesLeft[j]
				if min_col[col] == math.inf:
					matrix[row][col] = math.inf
				else:
					matrix[row][col] = matrix[row][col]- min_col[col]
		return bound



	def createMatrix (self, cities):
		myMatrix = [[math.inf for x in cities] for x in cities]
		for x in cities:
			for y in cities:
				if self._scenario._edge_exists[x._index][y._index]:
					first = cities[x._index]
					second = cities[y._index]
					pathcost = first.costTo(second)
					myMatrix[x._index][y._index] = pathcost
		return myMatrix
	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		

class node:
	def __init__(self,
				 current_city = None,
				 RCM = None,
				 bound = None,
				 remaining_cities = None,
				 path = None):
		self.current_city = current_city
		self.RCM = RCM
		self.bound = bound
		self.remaining_cities = remaining_cities
		self.path = path

	def __lt__(self,obj1):
		return self.bound < obj1.bound


