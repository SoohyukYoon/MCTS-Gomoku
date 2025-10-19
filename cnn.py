# Credits to CS 444: Deep Learning for Computer Vision for a lot of the framework for me to get started on this project
# i.e. what kind of objects were needed, which steps I needed to take to have a functioning CNN :>

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

# Torch shit
import torch  
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import transforms as T

# For Google Collab when using GPU for training 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#### DATASET #####
class GameDataset(Dataset): 
	"""
	Create class for Game Data: 
		Wrap into torch object and split into train and validation groups
		Dataset is the parent class of GameDataset, such that we are able to do; 
		batching, shuffling, and etc
	"""

	def __init__(self, root, split): 
		"""
		Args: 
			root (str): The root directory of the dataset
			split (str): Can be 'train', 'val', 'test'
			transform (callable, optional): Function used to data augment the board for more training
		"""
		self.root = root
		self.split = split
		with open(f"{root}/{split}.pkl", "rb") as f:  
			self.game_list = pickle.load(f)

	def __len__(self): 
		"""
		Returns: 
			The number of game states inside state_list
		"""
		return len(self.game_list)

	def __getitem__(self, idx): 
		"""
		Args: 
			idx (int): The index of the state-action point
		Returns: 
			state (Tensor): The current board state before action
			action (int): The action taken after the current board state
		"""
		states, action = self.game_list[idx]
		state_b = torch.tensor(states[0], dtype=torch.float32)
		state_w = torch.tensor(states[1], dtype=torch.float32)
		state_e = torch.tensor(states[2], dtype=torch.float32)
		action = torch.tensor(action, dtype=torch.long)

		# Cool thing: .array(): creates 3 seperate (225) tensors
		#			  .stck(): creates a single (3, 225) tensor
		return torch.stack([state_b, state_w, state_e]).reshape(3, 15, 15), action

def organize_games(root):
		"""
		Compiles the .xmlk data into state-action pairs for training: 
		 		True Label: action taken by human player
				Input: the state of the game before the action
					   the input is 3 dimensional, one-hot encoding, of black, white, and empty
		"""

		# Data on training input
		game_count = 0 
		state_count = 0 
		states_erased_count = 0 

		# Extract the entire file:
		tree = ET.parse(root)
		root = tree.getroot()
		game_set = set()
		game_list = []

		# Iterate through each game:
		k = 0 
		for game in root.findall('game'): 
			if k > 1: 
				break
			game_count += 1 
			# Initialize the features   
			game_state_b, game_state_w, game_state_e = [0] * 225, [0] * 225, [1] * 225

			# Get moves 
			moves = game.find('move').text.strip().split()
			# if k < 1: 
			# 	print("Moves: ", moves)
			# Since the game consists only of moves
			# iterate through the game, and create the
			# update game state per-iteration: 
			for i in range(len(moves) - 1): 
				state_count += 1 
				# Initialize current move 
				move = (int(moves[i][1:]) - 1) * 15 + ord(moves[i][0]) - ord('a')

				# 1) Since we need the action retrieve the next move 
				next_move = (int(moves[i + 1][1:]) - 1) * 15 + ord(moves[i + 1][0]) - ord('a')
				# if k < 1: 
				# 	print("Next move: ", next_move)
				# 	print("Row: ", moves[i + 1][1:])
				# 	print("Col: ", moves[i + 1][0])
				# 2) Update the game state with current move 
				game_state_b[move] = (i + 1) % 2 
				game_state_w[move] = i % 2 
				game_state_e[move] = 0 

				# 3) Check if game_state-next_move already in set 
				state_action = (hash(tuple((game_state_b + game_state_w + game_state_e))), next_move)
				if state_action not in game_set: 
					# Set it to hash to compress to single integer, faster processing during training
					# Must make to tuple since lists are mutable, so need to make to immutable for set
					game_set.add(state_action)
					game_list.append(([game_state_b.copy(), game_state_w.copy(), game_state_e.copy()], next_move))
				else: 
					states_erased_count += 1 
			k += 1
		# Print input data
		print(game_count)
		print(state_count)
		print(states_erased_count)

		# Create the 'dataset' directory if it doesn't exist
		import os
		if not os.path.exists("dataset"):
			os.makedirs("dataset")

		# Store these training datas inside training.pkl
		print("Game List Length: ", len(game_list))

		# Check data sanity
		print("\nData sanity check:")
		for i in range(min(5, len(game_list))):
			state, action = game_list[i]
			print(f"Example {i}: Action={action}, Board sum={sum(state[0]) + sum(state[1])}") # this was the culprit
			
		# Check label distribution
		actions = [x[1] for x in game_list]
		print(f"Unique actions: {len(set(actions))} out of {len(actions)} total")
		print(f"Action range: {min(actions)} to {max(actions)}")
		
		with open("dataset/training.pkl", "wb") as f: 
			pickle.dump(game_list[:10], f)
		with open("dataset/validation.pkl", "wb") as f: 
			pickle.dump(game_list[int(len(game_list) * (4/5)):], f)

# Generate GameDataset
organize_games('renjunet_v10_20180803.xml')

# Generate numpy dataset objects
train_dataset = GameDataset(root='dataset', split='training')
valid_dataset = GameDataset(root='dataset', split='validation')

# Test train_dataset: 
state, action = train_dataset.__getitem__(0)
# print("State after object creation: ", state)
# print("Action after object creation: ", action)


#### DATALOADER ####
"""
Wraps the Dataset with a DataLoader such that we are able to conviently: 
shuffle, batch, and multiprocessing(cpu, exclusively) our training data 
"""

# Initialize batch size, most papers use 32 
batch_size = 32 

# Shuffles our data-set to ensure randomization during each epoch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

sample_states, sample_actions = next(iter(train_loader))
# print(f"State batch shape: {sample_states.size()}")
# print(f"Action batch shape: {sample_actions.size()}")


#### CONVOLUTIONAL NEURAL NETWORKS BABY #### 
"""
I was planning on implementing Conv2d and MaxPool2d the way I did it on the 444 MP
But I realized my layer implementation are probably shit so decided not to do that lel
"""
class CNN(nn.Module): 
	"""
	Wrap the CNN in the torch nn module to inherit the module properties
	"""

	def __init__(self): 
		"""
		Initializes the CNN class with our layered routing
		Initializes the parent module, nn.Module
		"""
		# Calls the parent class, nn.Module for initialization 
		super().__init__()

		self.layers = nn.Sequential(
				# Layer 1: 5x5 Kernel
				nn.Conv2d(3, 128, 5, 1, 2),
				nn.ReLU(),

				# Layer 2: 3x3 Kernel 
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU(),

				# Layer 3: 3x3 Kernel 
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU(),

				# Layer 4: 3x3 Kernel 
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU(),

				# Layer 5: 3x3 Kernel 
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU(),

				# Layer 6: 1x1 Kernel 
				nn.Conv2d(128, 1, 1, 1, 0),

				# Flatten: Must do because CrossEntropy requires a single vector
				nn.Flatten()
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
		x = self.layers(x) 
		return x 

# Creat eht model instanel and move it to the GPU 
model = CNN().to(device)
print(model)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")

dummy_input = torch.randn(1, 3, 15, 15, device=device, dtype=torch.float)
output = model(dummy_input)
assert output.size() == (1, 225), f"Expected output size (1, 225), got {output.size()}!"
print("Test passed!")

#### Loss Function, Optimizer, and Scheduler #### 

# Define learning rate --- Remember gradient explosions need to be compensated with small LR
lr = 0.001

# Define gamma: learning rate decay, how quickly the training should converge to a loss
# We leave gamma to 0.9, such that learning rate is 90% of what is was last epoch
# TLDR; I mean soohyuk you already know this, but in case you are a dumbass to PREVEN OVERSHOOTING
gamma = 0.9

# Define Cross Entropy Loss
loss_f = nn.CrossEntropyLoss()

# Define Adam Optimizer: using this to mini-batch instead of SGD
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define Scheduler: Decays every epoch 
# Fun fact the "step" is called step, bc the change in LR changes like a step when plotted
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)


#### Training and Validation ####
def evaluate(model, data_loader): 
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
	with torch.no_grad(): 
		for states, actions in (data_loader): 
			states, actions = states.to(device), actions.to(device)
			output = model(states)
			# argmax at dim=1, which is 225, select one of 225
			pred = output.argmax(dim=1)
			# Uses item(): tensor_object -> int/float
			correct += (pred == actions).sum().item()
	accuracy = correct / len(data_loader.dataset)
	return accuracy

def train(model, optimizer, scheduler, n_epochs, loss_f):
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
	i = 0 
	for epoch in range(n_epochs): 
		# Sets the model to training mode: part of nn.Module
		#		We get the perks of automatic 1) dropout 2) batchnormalization, talked about in class but lowkey forget 
		#		Note: Either way even if not call .train() it gets called by default, but necessary
		#			  to call bc if we call .eval() then train again, eval removes dropout and batch normalization leading
		#			  to pretty shitty overfitted results.
		model.train()
		total_loss = 0 
		# After each epoch train_loader is reshuffled
		for (states, actions) in (train_loader): 
			# 0. Prepare data by moving it to device
			states, actions = states.to(device), actions.to(device)
			# 1. Clear previous Gradient, we don't want old gradient contributing again
			optimizer.zero_grad()
			# 2. Forward pass the states
			output = model(states)
			# 3. Calculate the loss
			loss = loss_f(output, actions)
			total_loss += loss
			# 4. Calculate all the Gradients, pytorch just does all this, it's like...magic
			loss.backward()
			# 5. Updates the weights with out Gradients, completing the backpropogation
			#		Note: loss_f and optimizer are different functions, but both share the same common parameters 
			optimizer.step()
		# 6. Update learning rate
		scheduler.step()
		# 7. Append training loss to plot dictionary
		history['train_loss'].append(total_loss.item() / max(1, len(train_loader)))
		# acc_v = evaluate(model, valid_loader)
		acc_t = evaluate(model, train_loader)
		# print(f"Epoch {epoch}, Valid Accuracy {acc_v * 100:.2f}%")
		history['train_acc'].append(acc_t)
		# print(f"Epoch {epoch}, Training Accuracy {acc_t * 100:.2f}%")
		i += 1 
	return history

n_epochs = 50
history = train(model, optimizer, scheduler, n_epochs, loss_f)

def plot_train_loss(history): 
	"""
	Plots the training loss 
	Args: 
		history: contains the loss list
	Return: 
		None --- call plt.show() to show the graph 
	"""
	# fig: entire canvas 
	# ax: actual plot where data is drawn 
	fig, ax = plt.subplots(figsize=(6,4))
	ax.plot(history.get('train_loss', []), label='train_loss')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Loss')
	ax.set_title('Training Curve')
	ax.legend()
	fig.tight_layout() # shows the labels I've defined
	return fig, ax

def plot_train_acc(history): 
	"""
	Plots the training loss 
	Args: 
		history: contains the loss list
	Return: 
		None --- call plt.show() to show the graph 
	"""
	# fig: entire canvas 
	# ax: actual plot where data is drawn 
	fig, ax = plt.subplots(figsize=(6,4))
	ax.plot(history.get('train_acc', []), label='train_acc')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Accuracy')
	ax.set_title('Training Curve')
	ax.legend()
	fig.tight_layout() # shows the labels I've defined
	return fig, ax

fig, ax = plot_train_loss(history)
fig_a, ax_a = plot_train_acc(history)
plt.show()




