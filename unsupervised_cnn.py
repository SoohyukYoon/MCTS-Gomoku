# Credits to CS 444: Deep Learning for Computer Vision for a lot of the framework for me to get started on this project
# i.e. what kind of objects were needed, which steps I needed to take to have a functioning CNN :>

"""
TODO: Formatting for comments have changed since I decided to embedd data type inside 
the function so I should probably get to that at some point since it looks 
pretty disgusting right now lel 
"""

from contextlib import nullcontext
import xml.etree.ElementTree as ET

import math 
import random 
import numpy as np 
import pickle 
from PIL import Image

# Used to make that nice percentage sign thing
from tqdm import tqdm

# Used to plot loss graphs
import matplotlib.pyplot as plt

# Torch shit -- most from 444
import torch  
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import transforms as T

# Torch shit for Multi-GPU training
import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os 

#### DATASET #####
class U_GameDataset(Dataset): 
	"""
	Create class for Game Data: 
		Wrap into torch object and split into train and validation groups
		Dataset is the parent class of GameDataset, such that we are able to do; 
		batching, shuffling, and etc
	"""

	def __init__(self, game_moves: torch.tensor): 
		"""
		In supervised_cnn.py I have a pickle file, but inside unsupervised_cnn.py
		I am going to directly input a list since I am only passing one game at a time.
		Args: 
			game_moves: Moves played 
		"""
		self.game_moves = game_moves 

	def __len__(self): 
		"""
		Returns: 
			The number of game states inside state_list
		"""
		return len(self.game_moves)

	def __getitem__(self, idx): 
		"""
		Args: 
			idx (int): The index of the state-action point
		Returns: 
			state (Tensor): The current board state before action
			action (int): The action taken after the current board state
		"""
		state, action, value = self.game_moves[idx]
		# Must be torch.long for cross_entropy, class based 
		# I also decided to specfically convert here since all other values are converted on 
		# initialization once, at the start of the simulation, but for action I constantly need 
		# to re-initialize to a torch.long because expand is a for loop on i -- slight optimization like a pro :>
		action = torch.tensor(action, dtype=torch.long) 
		value = torch.tensor(value, dtype=torch.float32)

		# Cool thing: .array(): creates 3 seperate (225) tensors
		#			  .stack(): creates a single (3, 225) tensor
		return state, action, value

#### DATALOADER ####
def u_prepare_dataloader(dataset: Dataset, rank: int): 
	return DataLoader(
		dataset,
		batch_size=8, # SGD since AlphaGo paper says "to minimize end-to-end evaluation time" but hopefully results are not so bad even with batch 1 
		# sampler handles the shuffling internally, good practice to not shuffle again
			# Why data gets corrupted makes no sense --- Gemini for 
		shuffle=False,
		# Include Distributed Sampler: Ensures that samples are chunked without overlapping samples
		# DDP_CHANGED
		sampler = DistributedSampler(dataset)
	)

#### CONVOLUTIONAL NEURAL NETWORKS #### 
class U_CNN(nn.Module): 
	"""
	Wrap the CNN in the torch nn module to inherit the module properties
	Note: BIG change from the original pure CNN implementation, we now branch the head 
	  	  to either the policy or value network. Oh my goddd can't believe I am actually 
	  	  implementing this :>>>
	"""
	def __init__(self, channels=128, layers=6): 
		"""
		Initializes the CNN class with our layered routing
		Initializes the parent module, nn.Module
		"""
		# Calls the parent class, nn.Module for initialization 
		super().__init__()

		self.backbone = nn.Sequential(
				# Layer 1: 5x5 Kernel
				nn.Conv2d(3, channels, 5, 1, 2, bias=False),
				nn.ReLU(),

				# Layer 2: 3x3 Kernel 
				nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
				nn.ReLU(),

				# Layer 3: 3x3 Kernel 
				nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
				nn.ReLU(),

				# Layer 4: 3x3 Kernel 
				nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
				nn.ReLU(),

				# Layer 5: 3x3 Kernel 
				nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
				nn.ReLU(),
		)

		self.policy_network = nn.Sequential(
			# Layer 6: 1x1 Kernel 
			nn.Conv2d(channels, 1, 1, 1, 0, bias=False),

			# Flatten: Must do because CrossEntropy requires a single vector
			nn.Flatten()
		)

		self.value_network = nn.Sequential(
			# Layer 6: 1x1 Kernel 
			# The expectation is that this number is between -1 and 1, so I need to figure something out
			nn.Conv2d(channels, 1, 15, 1, 0, bias=False)
		)

	def forward(self, x): 
		"""
		Forward pass for our CNN
		Applys the layers from __init__
		Args: 
			x (Tensor): The input tesnor of shape (N, 3, 225)
		Returns: 
			output (Tensor): The output tensor of shape (N, 225)
		"""
		output = self.backbone(x) 

		policies = self.policy_network(output)
		value = self.value_network(output)

		return policies, value  

#### TRAINING AND EVALUATION ####
class U_TRAIN(): 
	"""
	Defines the training class with the necessary eval and train function	
	
	Realize during training for selfplay the value is only either -1 or 1, and that is how we update it
	"""

	def __init__(self, model, lr, gamma, policy_criterion, value_criterion, optimizer, gpu_id=torch.device('cpu'), train_loader=None, valid_loader=None): 
		"""
		Initializes the class
		Args: 
			gpu_id: The GPUs ID 
			model: The model we are training with
			lr: learning rate
			gamma: learning rate decay 
			policy_criterion: loss function for policy network 
			value_criterion: loss function for value network
			optimizer: backpropogation policy
			gpu_id: ID of our GPU(s)
			train_loader: Training Dataset wrapped in Dataloader
			valid_loader: Validation Dataset wrapped in Dataloader
		"""
		self.gpu_id = gpu_id 
		# Wrap in DDP such that our trained model can be distributed across GPUs
			# device_ids: consists of a list of IDs the GPUs live on 
			# Since self.model refers to the DDP wrapped object we need to add .module to access model parameters
		# DDP_CHANGED -- 
		self.model = DDP(model.to(gpu_id), device_ids=[self.gpu_id]) 
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.lr = lr 
		# Define gamma: learning rate decay, how quickly the training should converge to a loss
		# We leave gamma to 0.9, such that learning rate is 90% of what is was last epoch --- prev overshooting
		self.gamma = gamma
		self.policy_criterion = policy_criterion 
		self.value_criterion = value_criterion
		self.optimizer = optimizer 
		# Define Scheduler: Decays every epoch 
		# Fun fact the "step" is called step, bc the change in LR changes like a step when plotted
		self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
		
	#### Training and Validation ####
	def evaluate(self, model, train_loader=False): 
		"""
		Args: 
			model (nn.Module/Unique Class): The specific model you've selected 
			data_loader (DataLoader): The DataSet object, to access the data  
		Returns: 
			accuracy (int/float): the accuracy of the model, based on number of correct labels over all test states
		"""
		# Without this I just get fucked 
		model.eval()

		# Apparently torch tracks gradients during forward pass, that takes some time
		# so taking that out is useful especially if we need to validate a million training data
		correct = 0 
		data_loader = self.valid_loader if not train_loader else self.train_loader
		with torch.no_grad(): 
			for states, actions, values in tqdm(data_loader): 
				states, actions = states.to(self.gpu_id), actions.to(self.gpu_id)
				policies, value = model(states)
				# argmax at dim=1, which is 225, select one of 225
				pred = policies.argmax(dim=1)
				# Uses item(): tensor_object -> int/float
				correct += (pred == actions).sum().item()
		accuracy = correct / len(data_loader.dataset)
		return accuracy

	def train(self, n_epochs):
		"""
		Trains the model given the provided dataset, aka updates the weights
		Args: 
			model (nn.Module/Unique Class): The specific model you've selected 
			optimizer (nn.your_favorite_optimizer): The specific optimizer you've selected
			scheduler (nn.your_favorite_scheduler): The specific scheduler you've selected
			n_epochs (int): The number of epochs you will train for
		"""
		# Create map for plotting loss
		history = {
				'train_loss': [],
				'train_acc': []
			}

		for epoch in (range(n_epochs)): 
			print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.train_loader.batch_size} | Steps: {len(self.train_loader)}")
			# Sets the model to training mode: part of nn.Module
			#		We get the perks of automatic 1) dropout 2) batchnormalization, talked about in class but lowkey forget 
			#		Note: Either way even if not call .train() it gets called by default, but necessary
			#			  to call bc if we call .eval() then train again, eval removes dropout and batch normalization leading
			#			  to pretty shitty overfitted results.
			# DDP_CHANGED
			self.model.train()
			total_loss = 0 
			# After each epoch train_loader is reshuffled
			for states, actions, values in self.train_loader: 
				# 0. Prepare data by moving it to GPU
				# Changed value from (5) to (5, 1, 1, 1) to keep loss.py happy
				states, actions, values = states.to(self.gpu_id), actions.to(self.gpu_id), values.view(-1, 1, 1, 1).to(self.gpu_id)
				# 1. Clear previous Gradient, we don't want old gradient contributing again
				self.optimizer.zero_grad()
				# 2. Forward pass the states
				# DDP_CHANGED
				out_actions, out_values = self.model(states)
				# 3. Calculate the loss
				#	actions does not need to be an indicator matrix, in torch merely providing the index is enough
				loss_policy = self.policy_criterion(out_actions, actions)
				loss_value = self.value_criterion(out_values, values)
				total_loss = loss_policy + loss_value
				# 4. Calculate all the Gradients, pytorch just does all this, it's like...magic
				# Tech can do loss_policy.backward() and loss_value.backward(), bc the gradients are independent. but this approach is cleaner
				total_loss.backward()
				# 5. Updates the weights with out Gradients, completing the backpropogation
				#		Note: criterion and optimizer are different functions, but both share the same common parameters 
				self.optimizer.step()
			# 6. Update learning rate
			self.scheduler.step()
			# 7. Append training loss to plot dictionary
			history['train_loss'].append(total_loss.item() / max(1, len(self.train_loader)))
			# acc_v = evaluate(model, valid_loader)
			use_train_loader = True 
			# acc_t = self.evaluate(self.model, use_train_loader)
			# print(f"Epoch {epoch}, Valid Accuracy {acc_v * 100:.2f}%")
			# history['train_acc'].append(acc_t)
			# print(f"Epoch {epoch}, Training Accuracy {acc_t * 100:.2f}%")

			# Save updated model to a file 
			if self.gpu_id == 0 and epoch % self.save_every == 0:
				self.save_checkpoint(epoch)

		return history

	def save_checkpoint(self, epoch): 	
		ckp = self.model.module.state_dict()
		PATH = "unsupervised_weights.pt"
		torch.save(ckp, PATH)
		print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
