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
		self.passNode = None

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

	def dynamic(self, matrix, cities):

		notInPath = [i for i in range(len(cities))]

		toBreak = False
		# find two things that point to each other
		for i in range(len(cities)):
			if toBreak == True:
				break

			for j in range(len(cities)):
				if j == i:
					continue

				if matrix[i][ j] != math.inf and matrix[j][ i] != math.inf:
					edgePath = [(i, j), (j, i)]  # this path is made up of the edges not the nodes
					nodePath = [i, j]
					notInPath.remove(i)
					notInPath.remove(j)
					toBreak = True
					break

		# for each node find what edge it matches too best and insert

		for node in notInPath:
			bestEdge = -1  # store the edge index in edge path
			bestCost = math.inf  # the best cost increase to add the node

			# find the edge that the node fits into best
			for edge in edgePath:
				originalCost = matrix[edge[0]] [edge[1]]
				newCost = matrix[edge[0]][ node] + matrix[node][ edge[1]]
				additionalCost = newCost - originalCost

				if additionalCost < bestCost:
					bestCost = additionalCost
					bestEdge = edge

			# put the node into the path of edges and node path
			if bestCost != math.inf:
				edgeIndex = edgePath.index(bestEdge)
				edgePath.insert(edgeIndex, (node, bestEdge[1]))
				edgePath.insert(edgeIndex, (bestEdge[0], node))
				del edgePath[edgeIndex + 2]
				nodePath.insert(edgeIndex + 1, node)
			else:
				notInPath.append(node)
				notInPath[notInPath.index(node)] = -1
		return nodePath

	def greedy( self,time_allowance=60.0 ):
		return self.greedy_wNode(time_allowance)
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
	
	
	def greedy2( self,time_allowance=60.0 ):
		start_time = time.time()
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)

		foundTour = False
		count = 0
		bssf = None

		masterMatrix = np.array(self.createMatrix(cities))
		path = [] # create a pathway


		# O: repeats the for loop worst case for each city as a new start so O(n*n^3) = O(n^4)
		# keep starting from new points intil solution is found
		while not foundTour:
			foundTour = True
			path = [] # reset the path

			# increment the starting spot
			currentSpot = count  # count starts at 0, just start looking from first city
			if currentSpot >= len(cities): # quit if we have cycled through all the spots
				foundTour = False
				break
			count +=1

			cityMatrix = np.array(copy.deepcopy(masterMatrix))  # create a copy
			cityMatrix[currentSpot, :] = math.inf
			path.append(currentSpot)  # add the first spot

			# O: finds the min of all the columns for each city so O(n*n^2)
			# for each city find the cheapest way to get to it
			for i in range(len(cities) - 1 ):
				# O: finds the minimum of every column in the matrix so O(n^2)
				min_index_column = np.argmin(cityMatrix, axis = 0) # this will return the index of the minimum values
				min_spot = min_index_column[currentSpot]

				# what if all are infinity? then the minumum spot will be infinity and need to restart
				if cityMatrix[min_spot][currentSpot] == math.inf:
					# if i != len(cities) - 1 :
					foundTour = False
					break

				cityMatrix[currentSpot][min_spot] = math.inf # inf out backtrace
				cityMatrix[:, currentSpot] = math.inf # we don't want to come back to the city again
				cityMatrix[min_spot, :] = math.inf
				currentSpot = min_spot # update current spot
				path.append(currentSpot) #add the new spot to the path

				if i == len(cities) - 2:
					if masterMatrix[path[0]][path[len(path) - 1]] == math.inf:
						foundTour = False


		# since the path is backwards because we found the best node to come from we need to reverse the path
		route = []
		i = ncities - 1
		while i >= 0:
			if i < len(path):
				route.append(cities[path[i]]) # this creates a route with the cities, we are adding the cities in the order of the path
			i -= 1


		# don't calculate the bssf if none was found
		if len(route) != 0:
			bssf = TSPSolution(route)

		end_time = time.time()


		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf if foundTour else None
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

	def greedy_wNode(self,time_allowance=60.0):
		start_time = time.time()
		results = {}
		default = self.defaultRandomTour()
		# greedy = self.greedy()
		# lowest_cost = greedy['cost']
		num_updates = 0
		path_not_found = True
		max_heap_len = 1
		pruned = 0
		num_sul = 0
		count = 0
		total_created = 0
		bssf = math.inf
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

		bound = self.reduceMatrix(mutationMatrix, currentCities)

		currentNode = node()
		currentNode.current_city = 0
		currentNode.bound = bound
		currentNode.path = [0]
		currentNode.RCM = copy.deepcopy(mutationMatrix)
		currentNode.remaining_cities = []

		for x in range(1, len(masterMatrix)):
			currentNode.remaining_cities.append(x)
		greedyNode = greedy_node()
		greedyNode.current_node = currentNode
		heapq.heappush(heap, greedyNode)

		while path_not_found and len(heap) and time.time() - start_time < time_allowance:
			greedyNode = heapq.heappop(heap)
			# print(currentNode.bound)
			if greedyNode.current_node.bound > bssf:
				pruned += 1

			else:
				for x in range(0, len(greedyNode.current_node.remaining_cities)):
					total_created += 1
					newnode = node()
					newnode.remaining_cities = []
					for y in range(0, len(greedyNode.current_node.remaining_cities)):
						if x != y:
							newnode.remaining_cities.append(greedyNode.current_node.remaining_cities[y])
						else:
							newnode.path = copy.deepcopy(greedyNode.current_node.path)
							newnode.path.append(greedyNode.current_node.remaining_cities[y])
							newnode.current_city = greedyNode.current_node.remaining_cities[y]
					newnode.bound = greedyNode.current_node.bound + greedyNode.current_node.RCM[greedyNode.current_node.current_city][
						greedyNode.current_node.remaining_cities[x]]
					newnode.RCM = self.chooseCity(greedyNode.current_node.RCM, greedyNode.current_node.current_city,
												  greedyNode.current_node.remaining_cities[x])
					if newnode.remaining_cities != 0:
						newnode.bound = newnode.bound + self.reduceMatrix(newnode.RCM, newnode.remaining_cities)

					if newnode.bound  == math.inf:
						pruned+=1
					else:
						if len(newnode.remaining_cities) == 0:
							num_sul += 1
							solution = newnode
							self.passNode = solution
							bssf = newnode.bound
							path_not_found = False
						else:
							newGreedyNode = greedy_node()
							newGreedyNode.current_node = newnode
							heapq.heappush(heap, newGreedyNode)
							if len(heap) > max_heap_len:
								max_heap_len = len(heap)
					count += 1
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
		results['cost'] = self.generateCost(solution.path) if foundTour else bssf
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
		return solution

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
		start_time = time.time()
		results = {}

		#greedy = self.greedy()
		#lowest_cost = greedy['cost']
		num_updates = 0
		max_heap_len = 1
		pruned = 0
		num_sul = 0
		count = 0
		total_created = 0


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
		dynamicNode = node()
		dynamicNode.path =  self.dynamic(masterMatrix,cities)

		dynamicNode.bound =self.generateCost(dynamicNode.path)

		bound = self.reduceMatrix(mutationMatrix,currentCities)

		currentNode = node()
		currentNode.current_city = 0
		currentNode.bound = bound
		bssf= self.two_opt(dynamicNode)
		#twoOptNode = self.two_opt(self.passNode)
		# math.inf

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
							solution=self.two_opt(newnode)
							bssf = solution.bound
							foundTour = True
						else:
							heapq.heappush(heap,newnode)
							if len(heap) > max_heap_len:
								max_heap_len = len(heap)
					count+=1
		end_time = time.time()
		results['cost'] = self.generateCost(solution.path) if foundTour else bssf
		results['time'] = end_time - start_time
		results['count'] = num_sul
		if foundTour:
			cityList = [cities[i] for i in solution.path]
			results['soln'] = TSPSolution(cityList)
		else:
			cityList = [cities[i] for i in dynamicNode.path]
			results['soln'] = TSPSolution(cityList)

		results['max'] = max_heap_len
		results['total'] = total_created
		results['pruned'] = pruned
		return results
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
		if type(obj1) == float:
			return self.bound < obj1
		return self.bound < obj1.bound


class greedy_node:
	def __init__(self,
				 current_node = None):
		self.current_node = current_node
	def __lt__(self,obj1):
		if len(self.current_node.path) != len(obj1.current_node.path):
			return len(self.current_node.path) > len(obj1.current_node.path)
		else:
			return self.current_node.bound < obj1.current_node.bound
