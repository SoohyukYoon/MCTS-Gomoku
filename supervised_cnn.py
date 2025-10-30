"""
This file is the exact same as a_cnn.py inside CNN folder. 
This file remains the same because in the supervised architecture we 
do not train the value network. The value network only is enforced 
when we are doing self play. 
"""

from contextlib import nullcontext
import xml.etree.ElementTree as ET
from urllib.parse import unquote


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


#### INITIALIZATION HELPERS #### 
def ddp_setup(rank, world_size): 
	"""
	Initialize distributed process group such that each process can communicate with each other
	Args: 
		world_size: total number of processes
		rank: unique identifier for a process, "usually" range 0--world_size-1
	"""
	# We initialize the Master: this machine coordinates communication with all processes
	# IP address of machine running rank 0 process, which is our machine --- so local 
	os.environ["MASTER_ADDR"] = "localhost"
	# Include free port on machine 
	os.environ["MASTER_PORT"] = "12344"
	# For CUDA GPU communications --- initialize default DDP group
	init_process_group(backend="nccl", rank=rank, world_size=world_size)

def organize_games(root, transform=True):
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
	for game in tqdm(root.findall('game')): 
		game_count += 1 
		# Initialize the features   
		game_state_b, game_state_w, game_state_e = np.array([0] * 225), np.array([0] * 225), np.array([1] * 225)

		# Get moves 
		raw = game.find('move').text.strip()
		moves = unquote(raw).split()
		# Since the game consists only of moves
		# iterate through the game, and create the
		# update game state per-iteration: 
		for i in range(-1, len(moves) - 1): 
			state_count += 1 

			if i == -1: 
				next_move = (int(moves[i + 1][1:]) - 1) * 15 + ord(moves[i + 1][0]) - ord('a')
			else: 
				# Initialize current move 
				move = (int(moves[i][1:]) - 1) * 15 + ord(moves[i][0]) - ord('a')

				# 1) Since we need the action retrieve the next move 
				next_move = (int(moves[i + 1][1:]) - 1) * 15 + ord(moves[i + 1][0]) - ord('a')

				# 2) Update the game state with current move 
				game_state_b[move] = (i + 1) % 2 
				game_state_w[move] = i % 2 
				game_state_e[move] = 0 

			# 3) Check if game_state-next_move already in set 
			game_state_b_copy = game_state_b.copy()
			game_state_w_copy = game_state_w.copy()
			game_state_e_copy = game_state_e.copy()

			# a) check for 0 rotation --- 
			combined = list(game_state_b_copy) + list(game_state_w_copy) + list(game_state_e_copy)
			state_action = (hash(tuple(combined)), next_move)
			if state_action not in game_set: 
				# Set it to hash to compress to single integer, faster processing during training
				# Must make to tuple since lists are mutable, so need to make to immutable for set
				game_set.add(state_action)
				game_list.append(([game_state_b_copy, game_state_w_copy, game_state_e_copy], next_move))
			else: 
				states_erased_count += 1 

			# for j in range(2): 
			# 	if j ==1: 
			# 		flip = True
			# 		game_state_b_copy = np.flipud(game_state_b_copy.reshape(15, 15)).flatten()
			# 		game_state_w_copy = np.flipud(game_state_w_copy.reshape(15, 15)).flatten()
			# 		game_state_e_copy = np.flipud(game_state_e_copy.reshape(15, 15)).flatten()
			# 		next_move = (14 - next_move // 15) * 15 + (next_move % 15)

			# 	# a) check for 0 rotation --- 
			# 	combined = list(game_state_b_copy) + list(game_state_w_copy) + list(game_state_e_copy)
			# 	state_action = (hash(tuple(combined)), next_move)
			# 	if state_action not in game_set: 
			# 		# Set it to hash to compress to single integer, faster processing during training
			# 		# Must make to tuple since lists are mutable, so need to make to immutable for set
			# 		game_set.add(state_action)
			# 		game_list.append(([game_state_b_copy, game_state_w_copy, game_state_e_copy], next_move))
			# 	else: 
			# 		states_erased_count += 1 

			# 	# b) check for 90 rotation
			# 	game_state_b_90 = np.rot90(game_state_b_copy.reshape(15, 15)).flatten() 
			# 	game_state_w_90 = np.rot90(game_state_w_copy.reshape(15, 15)).flatten() 
			# 	game_state_e_90 = np.rot90(game_state_e_copy.reshape(15, 15)).flatten()  
			# 	combined = list(game_state_b_90) + list(game_state_w_90) + list(game_state_e_90)
			# 	next_move_90 = (next_move // 15) + (14 - (next_move % 15)) * 15
			# 	state_action = (hash(tuple(combined)), next_move_90)
			# 	if state_action not in game_set: 
			# 		# Set it to hash to compress to single integer, faster processing during training
			# 		# Must make to tuple since lists are mutable, so need to make to immutable for set
			# 		game_set.add(state_action)
			# 		game_list.append(([game_state_b_90, game_state_w_90, game_state_e_90], next_move_90))
			# 	else: 
			# 		states_erased_count += 1 

			# 	# c) check for 180 rotation
			# 	game_state_b_180 = np.rot90(game_state_b_90.reshape(15, 15)).flatten() 
			# 	game_state_w_180 = np.rot90(game_state_w_90.reshape(15, 15)).flatten() 
			# 	game_state_e_180 = np.rot90(game_state_e_90.reshape(15, 15)).flatten() 
			# 	next_move_180 = (next_move_90 // 15) + (14 - (next_move_90 % 15)) * 15
			# 	combined = list(game_state_b_180) + list(game_state_w_180) + list(game_state_e_180)
			# 	state_action = (hash(tuple(combined)), next_move_180)
			# 	if state_action not in game_set: 
			# 		# Set it to hash to compress to single integer, faster processing during training
			# 		# Must make to tuple since lists are mutable, so need to make to immutable for set
			# 		game_set.add(state_action)
			# 		game_list.append(([game_state_b_180, game_state_w_180, game_state_e_180], next_move_180))
			# 	else: 
			# 		states_erased_count += 1 

			# 	# d) check for 270 rotation 
			# 	game_state_b_270 = np.rot90(game_state_b_180.reshape(15, 15)).flatten() 
			# 	game_state_w_270 = np.rot90(game_state_w_180.reshape(15, 15)).flatten() 
			# 	game_state_e_270 = np.rot90(game_state_e_180.reshape(15, 15)).flatten() 
			# 	next_move_270 = (next_move_180 // 15) + (14 - (next_move_180 % 15)) * 15
			# 	combined = list(game_state_b_270) + list(game_state_w_270) + list(game_state_e_270)
			# 	state_action = (hash(tuple(combined)), next_move_270)
			# 	if state_action not in game_set: 
			# 		# Set it to hash to compress to single integer, faster processing during training
			# 		# Must make to tuple since lists are mutable, so need to make to immutable for set
			# 		game_set.add(state_action)
			# 		game_list.append(([game_state_b_270, game_state_w_270, game_state_e_270], next_move_270))
			# 	else: 
			# 		states_erased_count += 1 

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
	# print("\nData sanity check:")
	# for i in range(min(5, len(game_list))):
	# 	state, action = game_list[i]
	# 	print(f"Example {i}: Action={action}, Board sum={sum(state[0]) + sum(state[1])}") # this was the culprit
		
	# Check label distribution
	actions = [x[1] for x in game_list]
	# print(f"Unique actions: {len(set(actions))} out of {len(actions)} total")
	# print(f"Action range: {min(actions)} to {max(actions)}")
	
	with open("dataset/training.pkl", "wb") as f: 
		pickle.dump(game_list[:int(len(game_list) * (4/5))], f)
	with open("dataset/validation.pkl", "wb") as f: 
		pickle.dump(game_list[int(len(game_list) * (4/5)):], f)

#### DATASET #####
class S_GameDataset(Dataset): 
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
		#			  .stack(): creates a single (3, 225) tensor
		return torch.stack([state_b, state_w, state_e]).reshape(3, 15, 15), action

#### DATALOADER ####
def s_prepare_dataloader(dataset: Dataset): 
	return DataLoader(
		dataset, 
		batch_size=32, 
		# sampler handles the shuffling internally, good practice to not shuffle again
			# Why data gets corrupted makes no sense --- Gemini for 
		shuffle=False,
		# Include Distributed Sampler: Ensures that samples are chunked without overlapping samples
		# DDP_CHANGED
		sampler = DistributedSampler(dataset)
	)

#### CONVOLUTIONAL NEURAL NETWORKS BABY #### 
"""
I was planning on implementing Conv2d and MaxPool2d the way I did it on the 444 MP
But I realized my layer implementation are probably shit so decided not to do that lel
"""
class S_CNN(nn.Module): 
	"""
	Wrap the CNN in the torch nn module to inherit the module properties
	"""

	def __init__(self, channels=128, layers=6): 
		"""
		Initializes the CNN class with our layered routing
		Initializes the parent module, nn.Module
		"""
		# Calls the parent class, nn.Module for initialization 
		super().__init__()

		self.layers = nn.Sequential(
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

				# Layer 6: 1x1 Kernel 
				nn.Conv2d(channels, 1, 1, 1, 0, bias=False),

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

#### TRAINING AND EVALUATION ####
class S_TRAIN(): 
	"""
	Defines the training class with the necessary eval and train function	
	"""

	def __init__(self, model, lr, gamma, criterion, optimizer, gpu_id, train_loader, valid_loader=None): 
		"""
		Initializes the class
		Args: 
			gpu_id: The GPUs ID 
			model: The model we are training with
			lr: learning rate
			gamma: learning rate decay 
			criterion: loss function 
			optimizer: backpropogation policy
			gpu_id: ID of our GPU(s)
			train_loader: Training Dataset wrapped in Dataloader
			valid_loader: Validation Dataset wrapped in Dataloader
		"""
		self.gpu_id = gpu_id 
		# Wrap in DDP such that our trained model can be distributed across GPUs
			# device_ids: consists of a list of IDs the GPUs live on 
			# Since self.model refers to the DDP wrapped object we need to add .module to access model parameters
		self.model = DDP(model.to(f"cuda:{gpu_id}"), device_ids=[self.gpu_id])
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.lr = lr 
		# Define gamma: learning rate decay, how quickly the training should converge to a loss
		# We leave gamma to 0.9, such that learning rate is 90% of what is was last epoch --- prev overshooting
		self.gamma = gamma
		self.criterion = criterion 
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
			for states, actions in tqdm(data_loader): 
				states, actions = states.to(self.gpu_id), actions.to(self.gpu_id)
				output = model(states)
				# argmax at dim=1, which is 225, select one of 225
				pred = output.argmax(dim=1)
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
		for epoch in tqdm(range(n_epochs)): 
			print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.train_loader.batch_size} | Steps: {len(self.train_loader)}")
			# Sets the model to training mode: part of nn.Module
			#		We get the perks of automatic 1) dropout 2) batchnormalization, talked about in class but lowkey forget 
			#		Note: Either way even if not call .train() it gets called by default, but necessary
			#			  to call bc if we call .eval() then train again, eval removes dropout and batch normalization leading
			#			  to pretty shitty overfitted results.
			self.model.train()
			total_loss = 0 
			# After each epoch train_loader is reshuffled
			for states, actions in self.train_loader: 
				# 0. Prepare data by moving it to GPU
				states, actions = states.to(self.gpu_id), actions.to(self.gpu_id)
				# 1. Clear previous Gradient, we don't want old gradient contributing again
				self.optimizer.zero_grad()
				# 2. Forward pass the states
				output = self.model(states)
				# 3. Calculate the loss
				#	actions does not need to be an indicator matrix, in torch merely providing the index is enough
				loss = self.criterion(output, actions)
				total_loss += loss
				# 4. Calculate all the Gradients, pytorch just does all this, it's like...magic
				loss.backward()
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
		PATH = "supervised_weights.pt"
		torch.save(ckp, PATH)
		print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

def plot_train_loss(history: list[float]): 
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

def plot_train_acc(history: list[float]): 
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
