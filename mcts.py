# Credit to GeeksForGeeks for the framework, though everything needed to be reworked 

from contextlib import nullcontext
import math 
import random 

class MCTSNode:
	def __init__(self, state, color=None, parent=None, action=None):
		self.color = color
		self.state = state                    # Current board: The state of the board
		self.parent = parent                  # Parent node: The state of board when curr player placed stone
		self.action = action                  # Move leading to this node: The move opp player made
		self.children = []                    # List of children: # of "next moves" we took, so far, since we find children randomly
		self.visits = 0                       # Visit count: # of times we visited this specific node
		self.wins = 0                         # Win count: # of times win when in this node
		self.untried_actions = self.get_actions()  # Available moves: Literally just iterates through empty slots 

	def get_actions(self):
		"""Return all empty cells."""
		return [i for i in range(49) if self.state[i] == '.']

	# Returns specific child if it exists, None else
	def get_child(self, child): 
		for i, child_ in enumerate(self.children): 
			if child.action == child_.action: 
				return child
		return None

	# Returns best child
	def best_child(self, c=0):
		"""Select child with best UCB1 score."""
		# print(self.children)
		if self.children == []:
			return None
		return max(self.children, key=lambda child:
				   (child.wins / child.visits) +
				   c * math.sqrt(math.log(self.visits) / child.visits))

	# Adds game played game state from the root
	def add_game(self, game, winner): 
		# If this is the OG Root
		if not self.parent: 
			self.visits += 1
		
		# Check if child node exists 
		game_best_child = game.best_child()
		if not game_best_child: 
			return 
		child = self.get_child(game_best_child)

		# Add child node if not exist
		if not child:
			child = self.expand(game_best_child.action)

		# Apply penalty/reward 
		if winner == 0:
			child.wins += 0.5 
		elif winner == child.color: 
			child.wins += 1
		else: 
			child.wins -= 2

		# Increment number of visits
		child.visits += 1

		# Recurse
		child.add_game(game_best_child, winner)

	def is_terminal(self):
		"""Check if the game has ended."""
		return self.check_winner() is not None or not self.get_actions()

	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	# Check for winner in Gomoku 
	def check_winner(self): 
		if self.action == None: 
			#print("No Self Action")
			return None
		dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
		player = 3 - self.get_current_player()
		# print("Player: ", player)
		# print("Curr Player: ", self.get_current_player())
		for dir in dirs: 
			winningCells = [[self.action // 7, self.action % 7]]
			for sgn in (1, -1): 
				rowStep = dir[0]
				colStep = dir[1]
				curRow = (self.action // 7) + rowStep * sgn
				# self.action[0] + rowStep * sgn
				curCol = (self.action % 7) + colStep * sgn
				# self.action[1] + colStep * sgn
				# print("Coords: ", curRow, curCol)
				if (curRow >= 7 or curRow < 0) or (curCol >= 7 or curCol < 0):
					continue
				while self.state[curRow * 7 + curCol] == player: 
					winningCells.append([curRow, curCol])
					if len(winningCells) > 4: 
						return None
					curRow += rowStep * sgn
					curCol += colStep * sgn
					if (curRow >= 7 or curRow < 0) or (curCol >= 7 or curCol < 0): 
						break 
			# print(len(winningCells))
			if len(winningCells) == 4: 
				# print("WINNER IS: ", player, "at ", self.action)
				return player
		return None
			
	def expand(self, action=None):
		"""Add one of the remaining actions as a child."""
		if not action: 
			random_index = random.randrange(len(self.untried_actions))
			action = self.untried_actions.pop(random_index)
		new_state = self.state[:]
		player = self.get_current_player()
		new_state[action] = player
		# print("ACTION: ", action)
		child = MCTSNode(new_state, 3 - self.color, parent=self, action=action)
		self.children.append(child)
		return child

	def get_current_player(self):
		"""Find whose turn it is."""
		b_count = self.state.count(1)
		w_count = self.state.count(2)
		return 1 if b_count == w_count else 2

	def rollout(self):
		"""Play random moves until the game ends."""
		state = self.state[:]
		player = self.get_current_player()

		while True:
			actions = [i for i in range(49) if state[i] == '.']
			if not actions: return 0.5  #Draw

			move = random.choice(actions)
			# print("rollout: ", move)
			state[move] = player
			player = 1 if player == 2 else 2

			winner = self.check_winner_for_state(state, self.color, move)
			if winner: return 1 if winner == 1 else 2

	def check_winner_for_state(self, state, color, action):
		"""Same winner check for rollout."""
		# print("check_winner_for_state: ", action)
		return MCTSNode(state,color,...,action).check_winner()

	def backpropagate(self, winner, loss):
		"""Update stats up the tree."""
		self.visits += 1
		# self.wins += result
		# if self.parent and self.parent.color == color:
		# 	self.parent.backpropagate(result, color)
		# elif self.parent: 
		# 	self.parent.parent.backpropagate(result, color)
		node = self 
		value = 10
		i = 1
		while node: 
			node.visits  += 1 
			player_at_parent = 3 - node.get_current_player()
			if winner == player_at_parent: 
				node.wins += 1 # / (i * 0.05) 
			elif winner == 0.5: 
				node.wins += 0.5
			else: 
				if loss: 
					node.wins -= 2
			node = node.parent
			i += 1 

def mcts_search(root, color, loss, iterations=500):
	# root = MCTSNode(root_state, color)
	i = 0 
	for _ in range(iterations):
		if i > 99998:
			print("count: ", i)
		node = root

		# Selection
		while not node.is_terminal() and node.is_fully_expanded():
			node = node.best_child(c=0)

		# Expansion
		if not node.is_terminal():
			node = node.expand()

		# Simulation: Does not actually creat MCT Nodes 
		winner = node.rollout()

		# Backpropagation: Update the root node 
		node.backpropagate(winner, loss)
		i += 1 
	root = root.best_child(c=0)

	#root.parent = None
	return root  # Return best move

# Combines two Game Trees resulted from training
def combine_game_trees(tree1, tree2):
	# Initialize resulting tree
	empty_board = ['.'] * 49 
	black = 1
	combined_tree = MCTSNode(empty_board, black)
	combined_tree.visits = tree1.visits + tree2.visits
	combined_tree.wins = tree1.wins + tree2.wins

	# Get children of each tree
	cmbnd_tr_children = combined_tree.children
	tree1_children = tree1.children 
	tree2_children = tree2.children 
	
	# Create child map of cmbnd tree
	combined_child_map = {}
	for child in tree1_children:
		if child.action not in combined_child_map: 
			combined_child_map[child.action] = [child]
		else: 
			combined_child_map[child.action].append(child)
	for child in tree2_children: 
		if child.action not in combined_child_map: 
			combined_child_map[child.action] = [child]
		else: 
			combined_child_map[child.action].append(child)

	# Combine action node together 
	for action, children in combined_child_map.items(): 
		# Merge the children 
		merged_child = combine_nodes_recursively(children)
		merged_child.parent = combined_tree
		# Append tree
		combined_tree.children.append(merged_child)

	return combined_tree

# Helper functions to combine MCTS sub-trees
def combine_nodes_recursively(children): 
	# Create merged child
	child = children[0]
	merged_child = MCTSNode(
		child.state[:], 
		child.color,
		action=child.action
		)

	# Sum statistics
	merged_child.visits = sum(child.visits for child in children)
	merged_child.wins = sum(child.wins for child in children)

	# Group grandchildren by action
	grandchildren_action_map = {}
	for child in children: 
		grandchildren = child.children
		for grandchild in grandchildren: 
			if grandchild.action in grandchildren_action_map: 
				grandchildren_action_map[grandchild.action].append(grandchild)
			else: 
				grandchildren_action_map[grandchild.action] = [grandchild]
	
	# Recurse on grandchildren 
	for action, grandchildren in grandchildren_action_map.items(): 
		merged_grandchild = combine_nodes_recursively(grandchildren)
		merged_grandchild.parent = merged_child
	
	return merged_child






