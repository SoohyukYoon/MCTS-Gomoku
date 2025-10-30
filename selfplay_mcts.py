# Reminder this is an action node, we do not have nodes for storing states

from typing import Any


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
	Custom class to handle MCTS operations
	Notice the name is "ActionNode" not "StateNode" since we are interested
	in the specific action take from a state
	"""
	def __init__(self, 
			parent: 'ActionNode'=None, 
			action: list[int]=None, 
			val: float=None,
			color: int=None,
			prob: float=None, 
			child_Ps: list[float]=None, 
			new_state: torch.Tensor=None, 
		):
		self.parent = parent			  			# Parent action node
		self.s = parent.new_s if parent else None	# Features before action
		self.a = action 						  	# Action taken from state s
		self.color = (parent.color + 1) % 2	if parent else color	# Color of action a 
		if parent: 								# Features after action a
			self.new_s = self.update_state(self.s, self.a, self.color)		 
		else:
			self.new_s =  new_state	
		self.v = val								# Valuation of our current state 
		self.child_Ps = child_Ps					# Probability vector of children 
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
		temp_new_s[2, action[0], action[1]] = 0
		return temp_new_s
			
	def get_bestchild(self): 
		"""
		Gets the best child based on the Custom UCB
		Returns: 
			best_child: ActionNode
		"""
		if self.children == []:
			return None
		# key: determines what metric to score child
		# lambda: creates anonymous function with parameter child and calculates Custom UCB
		return max(self.children, key=lambda child:
				   child.Q +
				   self.c * child.P * math.sqrt(self.N) / (1 + child.N))

	def expand(self): 
		"""
		If the current node does not have children create its children
		"""
		for i, p in enumerate(self.child_Ps):
			curr_state = self.new_s
			action = [i // 15, i % 15]
			next_color = (self.color + 1) % 2
			new_state = self.update_state(curr_state, action, next_color)

			self.children.append(ActionNode(
									parent=self, 
									action=action,  
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
		z = self.v
		while node: 
			node.N += 1 
			node.W += z
			node.Q = node.W / node.N
			z = -z
			node = node.parent

	def terminal(self): 
		"""Returns if the game has come to an end""" 
		return self.is_winner() or self.no_moves_left()
	
	def is_winner(self): 
		"""
		Checks if there is a winner in the game based on the current state, new_s, 
		by checking for a 5 in a row in the four possible configurations we can find it in
		Return: 
			boolean: True if there is winner, False if there is no winner
		"""
		# Edge case for root node without action
		if not self.a: 
			return None

		dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
		color = self.color

		for dir in dirs: 
			winningCells = [[self.a[0], self.a[1]]]
			for sgn in (1, -1): 
				rowStep = dir[0]
				colStep = dir[1]
				curRow = self.a[0] + rowStep * sgn
				curCol = self.a[1] + colStep * sgn
				if (curRow >= 15 or curRow < 0) or (curCol >= 15 or curCol < 0):
					continue
				while self.new_s[color, curRow, curCol] == 1: 
					winningCells.append([curRow, curCol])
					if len(winningCells) > 5: 
						return None
					curRow += rowStep * sgn
					curCol += colStep * sgn
					if (curRow >= 15 or curRow < 0) or (curCol >= 15 or curCol < 0): 
						break 
			if len(winningCells) == 5: 
				return True
		return False

	def no_moves_left(self): 
		"""
		Checks if there are no more moves left to play, 
		by checking that all the empty state board is 0.
		Return: 
			boolean: True if board is full, False is board is not full
		"""
		empty_state = self.new_s[2]
		return torch.all(empty_state == 0)

def mcts_search(model, root_state: torch.Tensor, color: int, simulations=1600):
	"""
	Does the custom MCTS search, rollouts are replaced with policies 
	Args: 
		model: Specific model that handles policy and value out to select moves 
		root_state: Initial board state for our search 
		simulations: Number of simulations before we select child
	Return: 
		selected_child: Child that was selected from the distribution 
	"""
	# Call CNN on root to get probs and val 
	probs, val = model(root_state)
	val = val.item()
	probs = get_legal_probs(probs.tolist()[0], root_state)

	# Covers both cases: Empty game or middle of game
	root = ActionNode(val=val, color=color, child_Ps=probs, new_state=root_state)
	# Back propogate on the root: important, else root(summation(N)) == 0, 
	# which makes the priors uninformative
	root.backpropogate()

	# Run by default 1600 simulations to decide which child to select
	for s in range(simulations): 
		node = root 

		# Get the leaf child
		while node.get_bestchild(): 
			node = node.get_bestchild()
		
		# Expand from leaf node 
		if node.terminal():
			terminal_routine(node, model)
			continue
		elif not node.child_Ps: 
			new_best_child_routine(node, model)
		else: 
			node.expand()

		# Select best child from this expansion 
		best_child = node.get_bestchild() 
			
		
		# Call CNN on the bestchild and update its parameters
		probs, val = model(best_child.new_s)
		probs = get_legal_probs(probs.tolist()[0], best_child.new_s)
		best_child.child_Ps = probs
		best_child.v = val.item()
	
		# Backpropogate on best_child 
		best_child.backpropogate()
	
	# Select a child from a distribution 
	child_index = select_child(generate_distribution(root.children))
	selected_child = root.children[child_index]
	select_child_action = (selected_child.a[0] * 15) + selected_child.a[1]

	# Return the child 
	return selected_child
		
def load_and_eval(model): 
	model.load_state_dict(torch.load('soogo_weights.pth', map_location=torch.device('cpu'))) 
	# Set to eval mode so that gradients don't flow back 
	model.eval()

def get_legal_probs(probs: list[float], state: torch.tensor): 
	"""
	Gets the legal move probabilities given board state 
	Args: 
		probs: Probability list generated from model 
		state: Current board state
	Returns: 
		probs: Updated list of probabilities with only legal moves 
	"""
	# .clamp(max=number) keeps element of tensor <= number 
	taken_board = (state[0] + state[1]).clamp(max=1)
	for r in range(15): 
		for c in range(15): 
			if taken_board[r, c] == 1: 
				# Set element to negative infinity if stone already present 
				probs[r * 15 + c] = -float('infinity')
	return probs

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

def terminal_routine(node, model):
	"""
	If node is the terminal node do not expand 
	and back prop
	Args: 
		node: Node we are backpropping from 
		mode: Model for executing policy and value operations
	""" 
	probs, val = model(node.new_s)
	node.v = val.item()
	node.backpropogate()

def new_best_child_routine(node, model): 
	"""
	
	Routine for when we have found a new best child 
	Args: 
		node: Node we are backpropping from 
		mode: Model for executing policy and value operations
	""" 
	probs, val = model(node.new_s)
	probs = get_legal_probs(probs.tolist()[0], node.new_s)
	node.child_Ps = probs
	node.v = val.item()
	node.backpropogate()
	node.expand()