import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 	
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable 

import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')


class GridL(nn.Module): 

	def __init__(self, env_infos, hidden, lr = 1e-3):
		nn.Module.__init__(self)
		self.linears = nn.ModuleList()
		for i in range(len(hidden)): 
			if i == 0: 
				l = nn.Linear(env_infos[0], hidden[0])
			else: 
				l = nn.Linear(hidden[i-1], hidden[i])
			self.linears.append(l)

		self.out = nn.Linear(hidden[-1], env_infos[1])
		self.adam = optim.Adam(self.parameters(), lr)

		self.use_dropout = False 


	def using_dropout(self, b): 

		self.use_dropout = b 

	def forward(self, x): 

		for l in self.linears: 
			x = F.relu(l(x))
			if self.use_dropout: 
				x = F.dropout(x,p = 0.1)
		# probs = F.log_softmax(self.out(x))
		out = self.out(x)
		return out 

	def think(self,x): 
		x.volatile = True
		deltas = self.forward(x).data.numpy().reshape(-1)
		return deltas

	def update(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

	def train(self, epochs, data, log = 150, show_loss = False ,name ='Bot',use_scheduler = False , step_size = 10):

		if use_scheduler:
			scheduler = StepLR(self.adam, step_size, gamma = 0.1)

		losses = []
		mean_loss = 0. 
		for epoch in range(epochs): 

			if(use_scheduler): 
				scheduler.step()

			for idx, (x,y) in enumerate(data): 

				xt = Variable(torch.Tensor(x))
				yt = Variable(torch.Tensor(y))

				pred = self.forward(xt)
				loss = F.mse_loss(pred, yt)
				
				mean_loss += loss.data[0]
				self.update(loss)

				

				if idx%log == 0: 
					mean_loss = mean_loss*1./log
					print('Epoch {}/{} - Batch {}/{} ({:.1f}%) \t Loss: {:.6f}'.format(epoch,
						epochs, idx*xt.shape[0], len(data.dataset), 
						xt.shape[0]*idx*100./len(data.dataset), mean_loss))

					if show_loss: 
						losses.append(mean_loss)
					mean_loss = 0. 

		if show_loss: 
			plt.plot(np.arange(len(losses)), losses, linewidth = 2)
			plt.xlabel('Iterations')
			plt.ylabel('Loss')
			plt.title('Loss over iterations')
			plt.pause(0.1)
			input('Press enter to continue')

		torch.save(self.state_dict(), name)

