"""
Unified training routine for supervised and unsupervised 
"""

from supervised_cnn import *
from unsupervised_cnn import * 
import torch  
from torch.utils.data import Dataset, DataLoader, ddp_set, prepare_dataloader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import transforms as T
import sys

#### SUPERVISED SECTION ####
def supervised_train(rank: int, world_size: int, total_epochs: int): 
	"""
	main training routing for supervised learning
	Args: 
		rank: id for our GPU 
		world_size: number og GPUs available
		total_epochs: number of epochs we train our data on 
	"""
	# Initialize DDP group
	ddp_setup(rank, world_size)
	# Organize game data: 
	organize_games('renjunet_v10_20180803.xml')
	# dataset, and wrap into dataloader
	train_loader = prepare_dataloader(GameDataset(root='dataset', split='training'))
	# Initialize model
	# Note: No .to(device), this I moved to training class initialization
	model = S_CNN()
	# Create Training instance
	train = S_TRAIN(
				rank,
				model, 
				lr=0.0001, 
				gamma=0.9, 
				optimizer=torch.optim.Adam(model.parameters(), 
				lr=0.0001), 
				criterion=nn.CrossEntropyLoss(), 
				train_loader=train_loader
			)	
	# Train the model 
	history = train.train(n_epochs=10)
	# Save the trained weights -- could u pickle, but pytorch was more sigma
	torch.save(model.state_dict(), 'supervised_weights.pth')
	# Destroy the process
	destroy_process_group()

#### UNSUPERVISED SECTION ####
def unsupervised_train(rank: int, world_size: int, total_epochs: int, total_games: int): 
	"""
	main training routing for unsupervised learning
	Args: 
		rank: id for our GPU 
		world_size: number og GPUs available
		total_epochs: number of epochs we train our data on 
	"""
	for g in range(total_games):
		# Initialize DDP group
		ddp_setup(rank, world_size)
		# Organize game data: 
		organize_games('renjunet_v10_20180803.xml')
		# dataset, and wrap into dataloader
		train_loader = prepare_dataloader(GameDataset(root='dataset', split='training'))
		# Initialize model
		# Note: No .to(device), this I moved to training class initialization
		model = U_CNN()
		# Create Training instance
		train = U_TRAIN(
					rank,
					model, 
					lr=0.0001, 
					gamma=0.9, 
					optimizer=torch.optim.Adam(model.parameters(), 
					lr=0.0001), 
					policy_criterion=nn.CrossEntropyLoss(), 
					value_criterion=nn.MSELoss(),
					train_loader=train_loader
				)	
		# Train the model 
		history = train.train(n_epochs=10)

	# Save the trained weights -- could u pickle, but pytorch was more sigma
	torch.save(model.state_dict(), 'unsupervised_weights.pth')
	# Destroy the process
	destroy_process_group()

if __name__ == "__main__": 
	"""
	Begins the training process
	"""
	# Check how many devices are available
	import sys
	world_size = torch.cuda.device_count()

	#### SUPERVISED SECTION ####
	"""
		mp.spawn: 
			1. Creates n processes, where each process is assigned to a GPU
			2. Each process gets a rank, id, range: 0,..., nprocs-1
		By design mp.spawn MUST call some main function
		By design it WILL pass rank as first arg, and args in order
	"""
	supervised_epochs = 10
	mp.spawn(supervised_train, args=(world_size, supervised_epochs), nprocs=world_size)

	#### UNSUPERVISED SECTION ####
	"""
		Design consideration: 
		1) during MCTS prob better to use CPU, but during CNN prob better to use GPU -- need to somehow fix this load issue 
		2) when creating U_CNN I also need to use ddp since that's going to be training
	"""
	unsupervised_epochs = 10 
	mp.spawn(unsupervised_train, args=(world_size, unsupervised_epochs), nprocs=world_size)


