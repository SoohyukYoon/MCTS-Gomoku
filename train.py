"""
Unified training routine for supervised and unsupervised 
"""

from supervised_cnn import *
from unsupervised_cnn import * 
from selfplay_mcts import *

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
	print("Beginning Supervised Training")
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
	print("Completed Supervised Training")

#### UNSUPERVISED SECTION ####
def unsupervised_train(total_epochs: int, total_games: int, rank: int=None, world_size: int=None): 
	"""
	main training routing for unsupervised learning
	Args: 
		rank: id for our GPU 
		world_size: number og GPUs available
		total_epochs: number of epochs we train our data on 
	"""
	print("Beginning Unupervised Training")
	# Initialize DDP group
	if rank and world_size: 
		ddp_setup(rank, world_size)

	model = load_supervised_weights(rank)

	# Create Training instance
	train = create_train_instance(rank, model)

	# Self play to train weights
	tot_history = selfplay_mcts(train, total_games)

	# Print the loss graph 
	plot_train_loss(tot_history)
	plt.show()

	# Save the trained weights -- could use pickle, but pytorch was more sigma
	torch.save(model.state_dict(), 'unsupervised_weights.pth')
	# Destroy the process
	if rank and world_size: 
		destroy_process_group()
	print("Completed Supervised Training")

#### HELPERS FUNCTIONS FOR UNSUPERVISED TRAINING ####
def load_supervised_weights(rank):
	"""
	Loads the supervised weights into the unsupervised models 
	Return: 
		model: Unsupervised model that has been updated with the supervised models weights

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
	# DDP_CHANGED : to.device('cpu') 
	device = torch.device('cpu')
	model = U_CNN().to(device)
	if rank: 
		supervised_state = torch.load('supervised_weights.pth', weights_only=True)
	else: 
		supervised_state = torch.load('supervised_weights.pth', weights_only=True, map_location=torch.device('cpu'))
	unsupervised_state = model.state_dict()

	# you NEED to make explicit the entire key, name.number.weight_or_bias
	for i in range(5): 
		backbone_key = f'backbone.{i*2}.weight' # get the conv layer key, at the even index bc ReLu
		supervised_key = f'layers.{i*2}.weight'  
		unsupervised_state[backbone_key] = supervised_state[supervised_key]

	unsupervised_state['policy_network.0.weight'] = supervised_state['layers.10.weight']

	# Load the modified state dict of U_CNN 
	model.load_state_dict(unsupervised_state)

	return model 

def create_train_instance(rank, model): 
	"""
	Creates training instance
	Args: 
		rank: Id of GPU, RIGHT NOW IT IS NOT USED FOR TESTING ON CPU
		model: Model we are training with 
	Return
		U_TRAIN
	"""
	return U_TRAIN(
				model, 
				lr=0.0001, 
				gamma=0.9, 
				policy_criterion=nn.CrossEntropyLoss(), 
				value_criterion=nn.MSELoss(), 
				optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
			)	

def selfplay_mcts(train: U_TRAIN=None, total_games: int=10000, rank: int=None): 
	"""
	The meat of where self play from MCTS occurs
	Args: 
		train: Train instance that has model and relevant functions for training
		tota_games: Total number of games we will train the model on 
	Return: 
		tot_history: Contains a list of the loss that has occured throughout slef-play training
	"""
	max_moves = 225
	tot_history = {
		'train_loss': []
	}
	# Loop over the total number of games we are going to train on 
	for g in range(total_games):

		# Set the model to eval to prevent gradients flowing back
		train.model.eval()
		# Initialize an empty board
		state_b = torch.zeros(15, 15)
		state_w = torch.zeros(15, 15)
		state_e = torch.ones(15, 15)
		state = torch.stack([state_b, state_w, state_e])

		# Initialize starting color to black and winner
		color = 1
		winner = -1

		# Initialize a game list that we will feed into training
		game_list = []

		# Loop over the moves being played in game g
		for m in (range(max_moves - 150)):
			child = mcts_search(train.model, state, color, simulations=10)
			state[child.color, child.a[0], child.a[1]] = 1 
			state[2, child.a[0], child.a[1]] = 0 
			color = (color + 1) % 2
			game_list.append([state, child.a[0] * 15 + child.a[1]])
			# print(f'move: {child.a[0] + 1}, {child.a[1] + 1}')
			print_game(state)
			if child.is_winner(): 
				print("game ended winner is: ", child.color)
				break 
			print()
		
		# Append values to the list
		append_value(game_list, winner)

		# Set the model to train
		train.model.train()
		# Update the dataloader to refresh with new U_MoveDataset
		train.train_loader = u_prepare_dataloader(U_GameDataset(game_list), rank)
		
		# From the moves played in the game train the model 
		history = train.train(n_epochs=10)
		tot_history['train_loss'].extend(history['train_loss'])
	return tot_history

def append_value(game_list, winner): 
	"""
	Appends the winner into each game move 
	Args: 
		game_list: list of moves played in the game
	"""
	value = 1 if winner != -1 else 0
	for i in range(len(game_list)): 
		game_list[i].append(value)
		value = -value

def print_game(state: torch.tensor): 
	for r in range(15): 
		print(r + 1, end='  ') if r < 9 else print(r + 1, end=' ')
		for c in range(15): 
			if state[0, r, c] == 1: 
				print('●', end=' ')
			elif state[1, r, c] == 1: 
				print('○', end=' ')
			else: 
				print('.', end=' ')
		print() 
	print('   a b c d e f g h i j k l m n o')
	print()

if __name__ == "__main__": 
	"""
	Begins the training process
	"""
	print("Training start")
	# Check how many devices are available
	import sys
	world_size = torch.cuda.device_count()
	print(f'world_size={world_size}')

	#### SUPERVISED SECTION ####
	"""
		mp.spawn: 
			1. Creates n processes, where each process is assigned to a GPU
			2. Each process gets a rank, id, range: 0,..., nprocs-1
		By design mp.spawn MUST call some main function
		By design it WILL pass rank as first arg, and args in order
	"""
	# supervised_epochs = 10
	# mp.spawn(supervised_train, args=(world_size, supervised_epochs), nprocs=world_size)

	#### UNSUPERVISED SECTION ####
	"""
		Design consideration: 
		1) during MCTS prob better to use CPU, but during CNN prob better to use GPU -- need to somehow fix this load issue 
		2) when creating U_CNN I also need to use ddp since that's going to be training
	"""
	unsupervised_epochs = 10 
	total_games = 1
	if world_size != 0: 
		mp.spawn(unsupervised_train, args=(world_size, unsupervised_epochs), nprocs=world_size)
	else: 
		unsupervised_train(unsupervised_epochs, total_games)
	
	print("Training complete")


