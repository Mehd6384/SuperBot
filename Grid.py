import numpy as np 
import pygame as pg 


def magnitude(v): 

	return np.sqrt(np.sum(v**2))

class Cell(): 

	def __init__(self, pos = np.array([0,0]), free = True): 

		self.pos = pos
		self.hCost = 0.
		self.gCost = 0.

		self.free = free
		self.parent = None

		self.voisins = []
	def fCost(self): 
		return self.hCost + self.gCost

	def getD(self, other): 

		dx = np.abs(self.pos[0] - other.pos[0])
		dy = np.abs(self.pos[1] - other.pos[1])

		if dx > dy :
			return 14*dy + (dx - dy)*10
		else : 
			return 14*dx + (dy - dx)*10

	def __repr__(self): 

		# return 'Cell at {} - G cost  = {} - H cost = {} '.format(self.pos,self.gCost, self.hCost)
		return 'Cell at {} - Free = {}'.format(self.pos, self.free)

class Grid(): 

	def __init__(self, nb_cell, obstacles): 

		self.nb_cell = nb_cell
		self.obstacles = obstacles
		self.inc = 1./nb_cell
		self.build_map() # creates self.map

		self.ready = False 
		# DEBUG 
		self.display_position = False  # see __repr__

	def build_map(self): 

		self.map = np.empty((self.nb_cell,self.nb_cell), dtype = object)

		inc = 1./self.nb_cell
		for i in range(self.nb_cell): 
			for j in range(self.nb_cell): 

				p = np.array([i*inc, j*inc])
				free = True
				for o in self.obstacles: 
					distance_to_obstacle = magnitude(o.position-p)
					if distance_to_obstacle < 2*inc: 
						free = False
				new_cell = Cell(p, free = free)
				self.map[i,j] = new_cell


		# FILLING NEIGHBOURS

		for i in range(self.nb_cell): 
			for j in range(self.nb_cell):  

				for a in range(-1,2): 
					for b in range(-1,2):
						if a+i >= 0 and a+i < self.nb_cell: 
							if b+j >= 0 and b+j < self.nb_cell:
								if not (a == 0 and b == 0): 
									self.map[i,j].voisins.append(self.map[i+a,j+b])

				# print('Case {}-{} has {} voisins'.format(i,j, len(self.map[i,j].voisins)))
				# input()

	def get_path(self, start, end):

		self.current_start = start 
		self.current_end = end
		self.current_chemin = self.solve_AStar(start, end)

		self.ready = True

		return self.current_chemin

	def draw(self, screen, screenSize): 


		sc = np.array(list(screenSize))
		for i in range(self.nb_cell): 
			for j in range(self.nb_cell): 

				current_pos = list(self.map[i,j].pos)
				current_pos[0] *= sc[0]
				current_pos[1] = (1- current_pos[1])*sc[1]

				size = int(self.inc*0.8*sc[0])   #### Careful ! If sc[0] != sc[1] cells will look weird
				color = (150,150,150) if self.map[i,j].free else (250,150,150)
				pg.draw.rect(screen, color, (current_pos[0], current_pos[1],size, size))

		self.draw_path(screen,screenSize)

	def draw_path(self, screen,screenSize): 

		sc = np.array(list(screenSize))
		# chemin = self.solve_AStar(start, end)
		if self.ready: 
			maxi = len(self.current_chemin)
			c1 = np.array([250,250,0])
			c2 = np.array([0,250,250])



			for i,cell in enumerate(self.current_chemin): 

				current_pos = list(cell)
				current_pos[0] *= sc[0]
				current_pos[1] = (1- current_pos[1])*sc[1]

				size = int(self.inc*0.8*sc[0])   #### Careful ! If sc[0] != sc[1] cells will look weird
				# color = (150,250,150) if (i!= 0 and i!=(len(chemin) -1)) else (150,150,250)

				c = i/maxi*(c2) + (1-(i/maxi))*c1

				pg.draw.rect(screen, c, (current_pos[0], current_pos[1],size, size))

	def solve_AStar(self, start,end): 


		cell_start,i,j = self.pos_to_cell(start, with_indicies = True)
		cell_end,ie,je = self.pos_to_cell(end, with_indicies = True)

		if not cell_end.free: 
			cell_end = self.find_closest_free(cell_end)


		chemin = []

		# print('Start point {} = {}'.format([i,j], cell_start))
		# print('End point {} = {}'.format([ie,je], cell_end))

		openSet = []
		closedSet = []
		over = False



		openSet.append(cell_start)
		# print('\n\n ****************** \n')
		while (len(openSet) > 0 or over == False): 

			if len(openSet) == 0: 

				break
			

			current_cell = openSet[0]

			for cell in openSet: 
				if cell.fCost() <= current_cell.fCost(): 
					current_cell = cell 

			openSet.remove(current_cell)
			closedSet.append(current_cell)

			if current_cell == cell_end: 
				over = True
				break


			for cell in current_cell.voisins: 
				if (cell in closedSet): 
					continue
				elif not(cell.free): 
					continue
				else: 
					nc = current_cell.getD(cell) + current_cell.gCost
					if ((nc < cell.gCost) or not(cell in openSet)): 
					#	print 'In last step'
						cell.gCost = nc
						cell.hCost = cell.getD(cell_end)
						cell.parent = current_cell
						if not (cell in openSet): 
							openSet.append(cell)



		path_cell = current_cell
		while (path_cell != cell_start): 
				chemin.append(path_cell.pos)
				path_cell = path_cell.parent

		chemin = chemin[::-1]
		return chemin


	def find_closest_free(self, cell): 

		for v in cell.voisins: 
			if v.free: 
				return v 

		D = 1000.
		best_i, best_j = 0,0
		x, y = 0,0
		for i in range(self.nb_cell): 
			for j in range(self.nb_cell): 
				if self.map[i,j].free: 
					d = magnitude(self.map[i,j].pos - cell.pos)
					if d < D: 
						D = d 
						best_i = i 
						best_j = j


		return self.map[best_i, best_j]


	def pos_to_cell(self, p, with_indicies = False): 

		i,j = p[0]/self.inc, p[1]/self.inc
		i = np.clip(i,0,self.nb_cell)
		j = np.clip(j,0,self.nb_cell)
		if with_indicies: 
			return self.map[int(i),int(j)],i,j
		else:
			return self.map[int(i),int(j)]

	def __repr__(self): 

		s = ''
		for i in range(self.nb_cell): 
			for j in range(self.nb_cell): 
				if self.display_position: 
					s += str(self.map[i,j].pos)
				else: 
					s += str(self.map[i,j].free) + ' '
			s += '\n'

		return s 

