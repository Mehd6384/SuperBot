import pygame as pg 
import numpy as np 
from Grid import Grid 

def normalize(v): 

	norm = magnitude(v)
	return v/norm

def magnitude(v): 

	m = np.sqrt(np.sum(v**2) + 1e-5)
	return m

class Solver(): 

	def __init__(self, robot, step = 1.):

		self.robot = robot
		self.nb_joints = self.robot.nb_joints
		self.step = step

	def compute_increment(self, direction, with_jaco = False ): 

		V = normalize(direction)#direction/(np.sqrt(np.sum(direction**2)) + 1e-5)		
		V = np.concatenate((V, [0]))
		joints_positions = self.robot.all_joints_positions()		
		
		#Get Jacobian 
		columns = []
		for i in range(self.nb_joints): 
			v = joints_positions[-1] - joints_positions[i]
			c = np.cross(np.array([0,0,1]), v)

			columns.append(c)	

		J = np.array(columns).T
	
		# Compute the delta angles 

		do = normalize(J.T.dot(V))

		if with_jaco: 
			return do*self.step, J
		else: 
			return do*self.step

	def get_only_jacobian(self): 

		joints_positions = self.robot.all_joints_positions()		
		
		#Get Jacobian 
		columns = []
		for i in range(self.nb_joints): 
			v = joints_positions[-1] - joints_positions[i]
			c = np.cross(np.array([0,0,1]), v)

			columns.append(c)	

		J = np.array(columns).T

		return J

	def compute_using_jaco(self, direction, jaco): 

		direction = np.concatenate((direction, [0]))
		V = normalize(direction).reshape(-1,1)

		do = normalize(jaco.T.dot(V)).reshape(-1)
		return do*self.step


class Robot(): 

	def __init__(self, nb_joints, length_joints, speed = 4, randomize = False): 

		self.nb_joints = nb_joints
		self.uLength = length_joints

		self.speed = speed
		self.angles = np.zeros((nb_joints))+1.

		self.solver = Solver(self)

		if randomize: 
			self.angles = np.random.uniform(low = -180., high = 180., size =(nb_joints))

		self.compute_positions()

	def rotate(self, action):  ## Old rotate method from actions

		jointIndex = action/2
		direction = 1 if action%2 == 0 else -1

		self.angles[int(jointIndex)] += direction*self.speed

	def rotate_ik(self, delta): # rotate from IK result
		# print('delta:{} '.format(delta))
		for i in range(len(self.angles)): 
			# print('Angle {} = {}'.format(i,self.angles[i]))
			self.angles[i] += delta[i]
			# print('Angle {} = {}'.format(i,self.angles[i]))

	def rotate_jaco(self, direction): 

		increment = self.solver.compute_increment(direction)

		# print(increment)
		for i in range(increment.shape[0]):
			self.angles[i] += increment[i]*self.speed

		return increment

	def rotate_jaco_and_jaco(self, direction): 

		increment, jaco = self.solver.compute_increment(direction, with_jaco = True)
		for i in range(increment.shape[0]): 
			self.angles[i] += increment[i]*self.speed

		return jaco

	def rotate_from_jaco(self, direction, jaco): 

		increment = self.solver.compute_using_jaco(direction, jaco)
		for i in range(len(self.angles)): #'''increment.shape[0]'''): 
			self.angles[i] += increment[i]*self.speed

		return jaco 

	def compute_positions(self): 
		self.normalize_angles()
		self.points = np.zeros((self.nb_joints+1,2))
		for i in range(self.nb_joints+1): 
			if i == 0: 
				self.points[i,:] = np.array([0.5,0.5])
			else:
				angle = (self.angles[i-1] + 90.)%360. 
				angle = np.radians(angle)
				self.points[i,:] = self.points[i-1,:] + self.uLength*np.array([np.cos(angle), np.sin(angle)])

	def normalize_angles(self): 
		for i in range(len(self.angles)): 

			self.angles[i] = (self.angles[i])%360.

	def joints_positions(self): 
		points = []
		for i,p in enumerate(self.points): 
			if i != 0: 
				points.append(p)	
		return points

	def all_joints_positions(self): 
		points = []
		for i,p in enumerate(self.points): 
			points.append(p)
		return points  

	def draw(self, screen, screenSize): 

		for j in range(self.nb_joints): 
			
			p0 = self.points[j].copy()
			p1 = self.points[j+1].copy()

			p0[0] = (p0[0]*screenSize[0])
			p0[1] = (screenSize[1]*(1-p0[1]))

			p1[0] = (p1[0]*screenSize[0])
			p1[1] = (screenSize[1]*(1-p1[1]))

			pg.draw.line(screen, (220,150,20), p0.astype(int), p1.astype(int), 5)
			pg.draw.circle(screen, (150,250,0), p0.astype(int), 10)
			pg.draw.circle(screen, (150,250,250), p1.astype(int), 10) 

class Target(): 

	def __init__(self, limits = [0.2,0.8,0.2,0.7], radius = 15):

	

		# margin_x = np.random.uniform(low = 0., high = 0.1)
		# margin_y = np.random.uniform(low = 0., high = 0.1)

		# case = np.random.randint(4)
		# if case == 0:
		# 	self.position = np.array([margin_x,margin_y])
		# elif case == 1: 
		# 	self.position = np.array([1-margin_x, margin_y])
		# elif case == 2: 
		# 	self.position = np.array([margin_x, 1-margin_y])
		# else: 
		# 	self.position = np.array([1-margin_x, 1-margin_y])


		x = np.random.uniform(low = limits[0], high = limits[1])
		y = np.random.uniform(low = limits[2], high = limits[3])
		self.position = np.array([x,y])
		

		self.radius = radius

	def draw(self, screen, screenSize): 

		pos = self.position.copy()
		pos[1] = screenSize[1]*(1-pos[1])
		pos[0] *= screenSize[0]
		

		pg.draw.circle(screen, (250,0,0), pos.astype(int), self.radius)
		pg.draw.circle(screen, (250,250,250), pos.astype(int), int(self.radius*2/3))
		pg.draw.circle(screen, (250,0,0), pos.astype(int), int(self.radius/3))

class Obstacle(): 

	def __init__(self, position, radius = 15): 

		self.position = position
		self.radius = radius

	def draw(self, screen, screenSize):

		pos = self.position.copy()
		pos[1] = screenSize[1]*(1-pos[1])
		pos[0] *= screenSize[0]

		tht = np.radians(90)
		direc = np.array([np.cos(tht), np.sin(tht)])
		d2 = np.array([-np.sin(tht), np.cos(tht)])

		pg.draw.circle(screen, (250,250,0), pos.astype(int), self.radius)
		pg.draw.line(screen, (0,0,0), (pos + self.radius*direc).astype(int), (pos - self.radius*direc).astype(int), 3)
		pg.draw.line(screen, (0,0,0), (pos + self.radius*d2).astype(int), (pos - self.radius*d2).astype(int), 3)

	def __repr__(self): 

		return 'Obstacle at {}'.format(self.position)

class World():

	def __init__(self, robot_joints = 2, joints_length = 0.25, robot_speed = 3,
				randomize_robot = False, randomize_target = False, reset_robot = False, 
				target_limits = [0.2,0.8,0.2,0.8], obstacles = 2, randomize_obstacles = False,
				max_steps = 200, grid_cell = 20, path_ready = False):

		self.randomize_robot = randomize_robot
		self.randomize_target = randomize_target
		self.target_limits = target_limits
		self.max_steps = max_steps
		self.robot_speed = robot_speed
		self.reset_robot = reset_robot
		self.randomize_obstacles = randomize_obstacles
		self.nb_obstacles = obstacles
		self.grid_cell = grid_cell

		self.robot = Robot(robot_joints,joints_length, speed = robot_speed, randomize = randomize_robot)
		self.target = Target(self.target_limits)
		self.obstacles = self.create_random_obstacles(obstacles)

		self.listed_positions = False
		self.robotParameters = [robot_joints, joints_length, robot_speed]

		self.steps = 0

		self.render_ready = False


		self.grid = Grid(self.grid_cell,self.obstacles)
		self.path_ready = path_ready
		
	def compute_path(self): 

		self.path = self.grid.get_path(self.robot.points[-1], self.target.position)
		self.path_ready = True

		# input(self.path)

	def setTargetPosition(self, pos): 
		#self.target.position = pos
		self.target_positions = pos
		self.target.position = np.array(pos[0])
		self.target_position_iterator = 0
		self.listed_positions = True

	def initRender(self, size = [700,700]):

		pg.init()
		self.screen = pg.display.set_mode(size)
		self.clock = pg.time.Clock()
		self.size = size

	def render(self, draw_path = False): 

		if not self.render_ready: 
			self.initRender()
			self.render_ready = True

		time = 30
		self.clock.tick(time)
		self.screen.fill((0,0,0))
		self.draw(self.screen, self.size, draw_path = draw_path)

		pg.display.flip() 

	def draw(self,screen, screenSize , draw_path = False): 

		if draw_path:
			self.grid.draw(screen,screenSize)

		self.robot.draw(screen, screenSize)
		self.target.draw(screen, screenSize)
		for o in self.obstacles: 
			o.draw(screen, screenSize)

		

	def create_random_obstacles(self, nb): 

		obstacles = []
		for i in range(nb): 
			p = np.random.uniform(0.,1.,(2))
			o = Obstacle(np.array(p))
			obstacles.append(o)

		return obstacles

	def set_obstacles_from_list(self, obstacles): 

		self.obstacles = obstacles
		self.grid = Grid(self.grid_cell, self.obstacles)
		self.compute_path()

	def set_obstacles_from_python_list(self, positions): 

		self.obstacles = []
		for p in positions: 
			o = Obstacle(np.array(p))
			self.obstacles.append(o)

		self.grid = Grid(self.grid_cell, self.obstacles)
		self.compute_path()

	def observe(self): 

		effectorPosition = self.robot.points[-1]
		vector = self.get_vector_target_effector()
		distance = magnitude(vector)

		state = []
		for a in self.robot.angles: 
			state.append(np.radians(a))

		for o in self.obstacles: 
			ob_to_eff = effectorPosition - o.position
			state.append(ob_to_eff[0])
			state.append(ob_to_eff[1])

		for v in vector: 
			state.append(v)

		complete = False

		if distance < 0.03:  # target reached
			complete = True
			success = 1

		contact = self.check_obstacles()
		if contact: 
			complete = True

		if self.steps > self.max_steps: 
			complete = True

		return state, complete

	def check_obstacles(self): 

		c = False
		eff_pos = self.robot.points[-1]
		for o in self.obstacles: 
			d = np.sqrt(np.sum((o.position - eff_pos)**2))
			if d < 0.03: c = True
		return c

	def get_vector_target_effector(self): 

		targetPosition = self.target.position
		effectorPosition = self.robot.points[-1]

		vector = targetPosition - effectorPosition
		return vector


	def step_deltas(self, delta): 

		self.robot.rotate_ik(delta)
		self.robot.compute_positions()

		self.steps += 1

		return self.observe()

	def step_real_vector(self, direction): 

		self.robot.rotate_jaco(direction)
		self.robot.compute_positions()
		self.steps += 1

		return self.observe()

	def step_real_vector_and_return_deltas(self, direction): 

		deltas = self.robot.rotate_jaco(direction)
		self.robot.compute_positions()
		self.steps += 1

		return [deltas, self.observe()]

	def step_gradient(self, max_distance = 0.2, obstacle_rep = 0.01): 

		vector_sum = np.zeros((2))
		eff_pos = self.robot.points[-1]
		direction = self.target.position - eff_pos

		attraction = 0.5*np.sum(direction**2)
		repulsion, directions_r = [], []

		for o in self.obstacles: 
			
			d = eff_pos - o.position
			m = magnitude(d)

			if m < max_distance:
				r = obstacle_rep*0.5*(0.1/m - 0.1/max_distance)**2.
			else: 
				r = 0


			vector_sum += normalize(d)*r

		vector_sum += direction*attraction
		vector_sum = normalize(vector_sum)

		data = self.step_real_vector_and_return_deltas(vector_sum)

		return data

	def step_astar(self): 

		if not self.path_ready: 
			self.compute_path()

		finishing = False 
		if len(self.path) == 0: 
			finishing = True
		if finishing: 
			current_target = self.target.position
			vector = current_target - self.robot.points[-1]
		else: 
			# print(len(self.path))
			current_target = self.path[0]
			vector = current_target - self.robot.points[-1]

			while magnitude(vector) < 0.05: 
				# print('Too close, removing closest pos')
				del self.path[0]
				if(len(self.path) == 0):
					break
				current_target = self.path[0]
				vector = current_target - self.robot.points[-1]

			 


		data = self.step_real_vector_and_return_deltas(vector)
		return data

	def get_robot_angles(self): 

		return self.robot.angles 

	def get_jaco_size(self): 

		return 2*self.robot.nb_joints

	def action_space_size(self): 

		return len(self.robot.angles)
		
	def observation_space_size(self): 

		return len(self.robot.angles) + 2*self.nb_obstacles + 2
		
	def get_env_infos(self): 
		return [self.observation_space_size(), self.action_space_size()]

	def reset(self): 

		self.steps = 0
		if self.reset_robot: 
			self.robot = Robot(self.robotParameters[0], 
						   self.robotParameters[1],
						   speed = self.robotParameters[2],
						   randomize= self.randomize_robot)


		if self.randomize_target: 
			self.target = Target(self.target_limits)
		if self.listed_positions: 
			self.target_position_iterator = (self.target_position_iterator+1)%len(self.target_positions)
			self.target.position = np.array(self.target_positions[self.target_position_iterator])

		if self.randomize_obstacles and self.nb_obstacles > 0:
			self.obstacles = self.create_random_obstacles(self.nb_obstacles)
			self.grid = Grid(self.grid_cell, self.obstacles)
		self.compute_path()

		state,_ = self.observe()
		return state

		# state,_,__,___ = self.observe()
		# return state

	def set_robot_reset(self): 

		self.robot = Robot(self.robotParameters[0], 
						   self.robotParameters[1],
						   speed = self.robotParameters[2],
						   randomize= self.randomize_robot)

	
	def close(self): 
		pg.quit()

# env = World(robot_joints = 2, joints_length = 0.25, robot_speed = 3,
# 			randomize_robot = True, randomize_target = True, reset_robot = False, 
# 			target_limits = [0.2,0.8,0.2,0.8], obstacles = 1, randomize_obstacles = False,
# 			max_steps = 200, grid_cell = 15)

# # env.setTargetPosition([[0.9,0.5]])
# env.set_obstacles_from_list([Obstacle(np.array([0.3,0.5]))])

# for i in range(103):
# 	s = env.reset()
# 	done = False 

# 	while not done: 
		
# 		data = env.step_astar()
# 		obs = data[1]
# 		s,done = obs
# 		env.render()
