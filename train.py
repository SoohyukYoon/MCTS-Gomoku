"""
Unified training routine for supervised and unsupervised 
"""

from supervised_cnn import *
from unsupervised_cnn import * 
from selfplay_mcts import *
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
	train_loader = s_prepare_dataloader(S_GameDataset(root='dataset', split='training'))
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
	# Initialize DDP group
	ddp_setup(rank, world_size)

	"""
	model.load_state_dict(torch.load('supervised_weights.pth', weights_only=True)) FAILS --- 
	since the model is not the same this is going to cause error -- need to manually set the dictionary.

	.state_dict(): maps the parameter names to their respective tensor values(the weights and biases). 
		Important note is this is not the actual architecture, rather it is the architecture in dictionary 
		form. We want to get this dictionary, such that we can modify its weights so that we can load 
		this dictionary back into the model to update its architecture using .load_state_dict()

		The returned dictionary has the following format, i.e. in the case of U_CNN(): 
		Update: Since I disabled bias in actuality there should be no '.bias', but 
		I am still going to keep the example below for generality
		{
			'backbone.0.weight': tensor([...]),
			'backbone.0.bias': tensor([...]),
			'backbone.2.weight': tensor([...]),
			'backbone.2.bias': tensor([...]),
			...
			'policy_network.0.weight: tensor([...]),
			'policy_network.0.bias: tensor([...]),
			'policy_network.1.weight: tensor([...]),
			'policy_network.1.bias: tensor([...]),
			'value_network.0.weight: tensor([...]),
			'value_network.0.bias: tensor([...]),
			'value_network.1.weight: tensor([...]),
			'value_network.1.bias: tensor([...])
		}
		Note: Notice that backbone excludes the odd layers, this is because it is ReLu --- state_dict only
			outputs learnable parameters, and since RELU is completely deterministic based on Conv it is excluded

			It's also weird I am even using a bias because I don't set any bias so I guess it's just adding 
			bullshit the entire time. Updated: Bias disabled

			No need to change anything for value_network since that is not part of the S_CNN
	"""
	# Note: No .to(device), this I moved to training class initialization
	model = U_CNN()
	supervised_state = torch.load('supervised_weights.pth', weights_only=True)
	unsupervised_state = model.state_dict()

	# By default just doing network.number without weight or bias implies both, 
	# in our case we have no bias, but it still works
	for i in range(5): 
		backbone_key = f'backbone.{i*2}' # get the conv layer key, at the even index bc ReLu
		supervised_key = f'layer.{i*2}'  
		unsupervised_state[backbone_key] = supervised_key

	# Here I make .weight explicit just as sake of example
	unsupervised_state['supervised_network.0.weight'] = supervised_state['layer.11.weight']

	# Load the modified state dict of U_CNN 
	model.load_state_dict(unsupervised_state)

	# Create Training instance
	train = U_TRAIN(
				rank,
				model, 
				lr=0.0001, 
				gamma=0.9, 
				optimizer=torch.optim.Adam(model.parameters(), 
				lr=0.0001), 
				policy_criterion=nn.CrossEntropyLoss(), 
				value_criterion=nn.MSELoss()
			)	

	"""
	The meat of where everything comes together
	"""
	max_moves = 225
	tot_history = []
	# Loop over the total number of games we are going to train on 
	for g in range(total_games):
		# Initialize an empty board
		state_b = torch.zeros(15, 15)
		state_w = torch.zeros(15, 15)
		state_e = torch.ones(15, 15)
		state = torch.stack([state_b, state_w, state_e])

		# Initialize starting color to black 
		color = 0

		# Initialize a game list that we will feed into training
		game_list = []

		# Loop over the moves being played in game g
		for m in range(max_moves):
			child = mcts_search(model, state, color, simulations=1600)
			state[color, child.action // 15, child.action % 15] = 1 
			state[2, child.action // 15, child.action % 15] = 0 
			color = (color + 1) % 2
			game_list.append([state, child.action])
			if child.is_winner(): 
				# Append value to each move 
				append_value(game_list) 
				break 

		# Update the dataloader to refresh with new U_MoveDataset
		train.train_loader = prepare_dataloader(game_list)
		
		# From the moves played in the game train the model 
		history = train.train(n_epochs=10)
		tot_history.append(history)

	# Print the loss graph 
	plot_train_loss(tot_history)

	# Save the trained weights -- could use pickle, but pytorch was more sigma
	torch.save(model.state_dict(), 'unsupervised_weights.pth')
	# Destroy the process
	destroy_process_group()

def append_value(game_list): 
	value = 1
	for i in range(len(game_list)): 
		game_list[i].append(value)
		value = -value

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


