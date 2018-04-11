from GridReacher import World 
import numpy as np 
import matplotlib.pyplot as plt 
from GridLearner import GridL 
from GridData import GridData

import torch
from torch.autograd import Variable

env = World(robot_joints = 3, joints_length = 0.2, robot_speed = 3,
			randomize_robot = True, randomize_target = True, reset_robot = False, 
			target_limits = [0.2,0.8,0.2,0.8], obstacles = 3, randomize_obstacles = False,
			max_steps = 200, grid_cell = 15)

env.set_obstacles_from_python_list([[0.3,0.5], [0.7,0.2], [0.6,0.7]])

train = False  
if train: 
	loader, infos = GridData(env, 10000)

	epochs = 50
	bot = GridL(infos,[128,64])
	bot.using_dropout(True)
	bot.train(epochs, loader, show_loss = True, use_scheduler = True, step_size = 20, name = 'BotDropout')
else: 
	loader, infos = GridData(env, 1)
	bot = GridL(infos,[128,64])
	bot.load_state_dict(torch.load('BotDropout'))

env.set_robot_reset()
env.render()

input('Ready to test')
for epoch in range(100): 

	s = env.reset()
	done = False 

	while not done: 

		env.render(draw_path = True)
		
		# noise = np.random.normal(0,0.05, (len(s)))
		# s = np.array(s) + noise
		# deltas = bot.think(Variable(torch.from_numpy(s.reshape(1,-1)).float()))

		deltas = bot.think(Variable(torch.Tensor(s)).unsqueeze(0))
		print('Deltas: {} \n Angles {}'.format(deltas, env.get_robot_angles()))
		s,done = env.step_deltas(deltas*3)