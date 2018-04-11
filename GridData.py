import pygame as pg
import numpy as np 
import pickle
from GridReacher import World

import torch
from torch.utils.data import TensorDataset, DataLoader


class BotData(): 

	def __init__(self, env): 

		self.env = env 
		self.env_infos = self.env.get_env_infos()

	def generate(self, nb_trajectoires, log_freq = 10): 

		print('Starting data creation...')

		x = []
		y = []
		log = nb_trajectoires/log_freq
		for epoch in range(nb_trajectoires): 

			s = self.env.reset()
			done = False

			while not done:

				data = self.env.step_astar()
				deltas, obs = data[0], data[1]

				# s = np.array(s)
				# s += np.random.normal(scale = 0.1, size = (s.shape))

				x.append(s)
				y.append(deltas)

				s, done = obs 

			if epoch%log == 0 and epoch > 0: 
				print('Dataset at \t {:.1f}%'.format(epoch*100./nb_trajectoires))


		data = [x,y]
		infos = [nb_trajectoires, self.env_infos[0], self.env_infos[1]]
		return data, infos 

class Dataset(): 

	def __init__(self, data, infos, batch_size = 32): 

		self.data = data
		self.infos = infos
		self.batch_size = batch_size

	def get_inout_sizes(self): 

		return self.infos[1], self.infos[2]

	def ToTensor(self): 

		x = torch.Tensor(self.data[0])
		y = torch.Tensor(self.data[1])

		print('Dataset of size: {} ({} Trajectories) -- Input size {} -- Output size {}'.format(
			x.shape[0], self.infos[0], self.infos[1], self.infos[2]))

		return TensorDataset(x,y)

	def ToLoader(self):

		td = self.ToTensor()
		data = DataLoader(td, shuffle = True, batch_size = self.batch_size)

		return data

class GridData():

	def __new__(self, env, nb_t, loader_batch_size = 32): 

		self.env = env
		dc = BotData(self.env)
		data, infos = dc.generate(nb_t)
		self.loader = Dataset(data, infos, loader_batch_size).ToLoader()

		return self.loader, [infos[1],infos[2]]


# env = World(robot_joints = 2, joints_length = 0.25, robot_speed = 3,
# 			randomize_robot = True, randomize_target = True, reset_robot = False, 
# 			target_limits = [0.2,0.8,0.2,0.8], obstacles = 1, randomize_obstacles = False,
# 			max_steps = 200, grid_cell = 15)

# env.set_obstacles_from_python_list([[0.3,0.5]])

# d, dd = GridData(env, 1000)

# for i in range(103):
# 	s = env.reset()
# 	done = False 

# 	while not done: 
		
# 		data = env.step_astar()
# 		obs = data[1]
# 		s,done = obs
# 		env.render()