# Credit to GeeksForGeeks for the framework, though everything needed to be reworked 
# Reminder this is an action node, we do not have nodes for storing states

from typing import Any


from soogo_cnn import CNN

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


class ActionNode:
	"""
	Custom class to handle MCTS operations. 
	Notice the name is "ActionNode" not "StateNode" since we are interested
	in the specific action take from a state
	"""
	def __init__(self, 
			parent: 'ActionNode'=None, 
			action: tuple[int, int]=None, 
			val: torch.float32=None,
			prob: torch.float32=None, 
			new_state: torch.Tensor=None, 
		):
		self.parent = parent			  			# Parent action node
		self.s = parent.new_s	 			 		# Features before action
		self.a = action 						  	# Action taken from state s
		self.color = (parent.color + 1) % 2			# Color of action a 
		if new_state: 								# Features after action a
			self.new_s = new_state 
		else:
			self.new_s =  self.update_state(self.s, self.a, self.color)		
		self.v = val								# Valuation of our current state 
		self.child_Ps = None						# Probability vector of children 
		self.P = prob						  		# Probability of chosing action a from action s 
		self.W = 0							  		# Total value of taking action a from state s
		self.N = 0						  			# Total number of times went from state s by action a 
		self.Q = 0									# Average value: self.W / self.N
		self.c = 1									# Constant that balances exploitation and exploration
		self.children = []					  		# Children action nodes from state s' --- the state action a lands on

	def update_state(self, state, action, color): 
		"""
		Args: 
			state: The state you want to update
			action: The location you want to add new stone to state
			color: The color of the stone being placed, color of action
		Return:
			temp_new_s: newly updated state after taking action a from state s 
		"""
		# .clone() for pytorch tensors
		temp_new_s = state.clone()
		temp_new_s[color, action[0], action[1]] = 1 
		temp_new_s[(color + 1)%2, action[0], action[1]] = 0
		return temp_new_s
			
	def get_bestchild(self): 
		"""
		Gets the best child based on the Custom UCB
		Returns: 
			best_child: ActionNode
		"""
		if self.children == []:
			print("No children")
			return None
		# key: determines what metric to score child
		# lambda: creates anonymous function with parameter child and calculates Custom UCB
		return max(self.children, key=lambda child:
				   child.Q +
				   self.c * child.P * math.sqrt(self.N) / (1 + child.V))

	def expand(self): 
		"""
		If the current node does not have children create its children
		"""
		for i, p in enumerate(self.child_Ps):
			curr_state = self.new_state
			action = tuple(i // 15, i % 5)
			next_color = (self.color + 1) % 2
			new_state = self.update_state(curr_state, action, next_color)

			self.children.append(ActionNode(
									parent=self, 
									action=tuple(i // 15, i % 5),  
									prob=p, 
									new_state=new_state
									)
								)

	def backpropogate(self): 
		"""
		Propogates leaf valuation from the leaf 
		up to the root.
		"""
		node = self
		z = self.val
		while node: 
			node.N += 1 
			node.W += z
			node.Q = node.W / node.N
			z = -z


# Initialize our device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def mcts_search(root_state: torch.Tensor, color: int, simulations=1600):
	# Load the trained weights and set to eval
	model = load_and_eval(CNN().to(device))

	# Call CNN on root to get probs and val 
	probs, val = model(root_state)

	# Covers both cases: Empty game or middle of game
	root = ActionNode(val=val, child_Ps=probs, new_state=root_state)
	# Back propogate on the root: important, else root(summation(N)) == 0, 
	# which makes the priors uninformative
	root.backpropogate()

	# Run by default 1600 simulations to decide which child to select
	for s in simulations: 
		node = root 

		# Get the leaf child
		best_child = node.get_bestchild()
		while best_child: 
			node = best_child
			best_child = best_child.get_bestchild()
		
		# Expand from best_child 
		node.expand()

		# Select best child from this expansion 
		best_child = node.get_bestchild() 
		
		# Call CNN on the bestchild and update its parameters
		probs, val = model(best_child)
		best_child.val = val 
		best_child.child_Ps = probs

		# Backpropogate on best_child 
		best_child.backpropogate()
	
	# Select a child from a distribution 
	child_index = select_child(generate_distribution(root.children))

	# Return the child 
	return root.children[child_index]
		
def load_and_eval(model): 
	print("Model initialized!")
	model.load_state_dict(torch.load('soogo_weights.pth', map_location=torch.device(device))) 
	# Set to eval mode so that gradients don't flow back 
	model.eval()

def generate_distribution(children): 
	"""
	Generate a distribution of likelihoods a child will be selected
	from the visit count of each child 
	Args: 
		children: children of the root
	Return: 
		dist: a list that represents the distribution of an action 
	"""
	# Place holder, T should be an input determined inside Training 
	T = 1 

	# Get the terms: 
	# N(s, a_i)^1/T and summation(N(s, a_i)^1/T)
	roots = []
	tot = 0 
	for child in children: 
		T_root = child.N ** (1/T)
		tot += T_root
		roots.append(T_root)
	
	# Create the distribution 
	dist = [] 
	curr_intrvl = 0 
	for r in roots: 
		curr_intrvl += r / tot
		dist.append(curr_intrvl)

	return dist

def select_child(dist): 
	"""
	Generate random number between 0 and 1,
	then Binary Search to find child index
	Args: 
		dist: a list that represents the distribution of an action 
	Return: 
		the index of the action that the rand_numbr landed on
	"""
	# Default generates a number between 0 and 1 
	rand_numbr = random.random()

	# BST: Return r since (dist[l], dist[r]] is our interval 
	# Regular BST also beautifully handles cases where dist[l] == dist[r]
	# since if that occurs than the only time a return from that condition can occur 
	# inside the while loop 
	r, l = len(dist) - 1, 0
	while r - l > 1:
		m = (l + r) // 2  
		if rand_numbr < dist[m]:	
			r = m  
		elif rand_numbr > dist[m]: 
			l = m 
		else: 
			return m 
	return r 



	