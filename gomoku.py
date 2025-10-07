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
		return [(i, j) for i in range(9) for j in range(9) if self.state[i][j] == '.']

	def is_terminal(self):
		"""Check if the game has ended."""
		return self.check_winner() is not None or not self.get_actions()

	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	# Check for winner in Gomoku 
	def check_winner(self): 
		# print('\n')
		# print("CHECKING FOR WINNERS")
		# print("ACTION: ", self.action)
		# for row in self.state: 
		# 	for ele in row: 
		# 		if ele == 1: 
		# 			print('●', end=' ')
		# 		if ele == 2: 
		# 			print('○', end=' ')
		# 		if ele == '.':
		# 			print('.', end=' ')
		# 	print() 

		if self.action == None: 
			#print("No Self Action")
			return None
		dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
		player = 3 - self.get_current_player()
		# print("Player: ", player)
		# print("Curr Player: ", self.get_current_player())
		for dir in dirs: 
			winningCells = [[self.action[0], self.action[1]]]
			for sgn in (1, -1): 
				rowStep = dir[0]
				colStep = dir[1]
				curRow = self.action[0] + rowStep * sgn
				curCol = self.action[1] + colStep * sgn
				# print("Coords: ", curRow, curCol)
				if (curRow > 8 or curRow < 0) or (curCol > 8 or curCol < 0):
					continue
				while self.state[curRow][curCol] == player: 
					winningCells.append([curRow, curCol])
					if len(winningCells) > 5: 
						return None
					curRow += rowStep * sgn
					curCol += colStep * sgn
					if (curRow > 8 or curRow < 0) or (curCol > 8 or curCol < 0): 
						break 
			# print(len(winningCells))
			if len(winningCells) == 5: 
				# print("WINNER IS: ", player, "at ", self.action)
				return player
		return None
			
	def expand(self):
		"""Add one of the remaining actions as a child."""
		random_index = random.randrange(len(self.untried_actions))
		action = self.untried_actions.pop(random_index)
		new_state = [row[:] for row in self.state]
		player = self.get_current_player()
		new_state[action[0]][action[1]] = player
		# print("ACTION: ", action)
		child = MCTSNode(new_state, 3 - self.color, parent=self, action=action)
		self.children.append(child)
		return child

	def get_current_player(self):
		"""Find whose turn it is."""
		b_count = sum(row.count(1) for row in self.state)
		w_count = sum(row.count(2) for row in self.state)
		return 1 if b_count == w_count else 2

	def best_child(self, c=1.4):
		"""Select child with best UCB1 score."""
		return max(self.children, key=lambda child:
				   (child.wins / child.visits) +
				   c * math.sqrt(math.log(self.visits) / child.visits))

	def rollout(self):
		"""Play random moves until the game ends."""
		state = [row[:] for row in self.state]
		player = self.get_current_player()

		while True:
			actions = [(i, j) for i in range(9) for j in range(9) if state[i][j] == '.']
			if not actions: return 0.5  # Draw

			move = random.choice(actions)
			# print("rollout: ", move)
			state[move[0]][move[1]] = player
			player = 1 if player == 2 else 2

			winner = self.check_winner_for_state(state, self.color, move)
			if winner: return 1 if winner == 1 else 2

	def check_winner_for_state(self, state, color, action):
		"""Same winner check for rollout."""
		# print("check_winner_for_state: ", action)
		return MCTSNode(state,color,...,action).check_winner()

	def backpropagate(self, result, color):
		"""Update stats up the tree."""
		self.visits += 1
		self.wins += result
		if self.parent and self.parent.color == color:
			self.parent.backpropagate(result, color)
		elif self.parent: 
			self.parent.parent.backpropagate(result, color)


def mcts_search(root_state, color, iterations=500):
	root = MCTSNode(root_state, color)

	for _ in range(iterations):
		if _ > 1000 and _ % 1000 == 0:
			print((_ / 1000)*6) 
		node = root

		# Selection
		while not node.is_terminal() and node.is_fully_expanded():
			node = node.best_child()

		# Expansion
		if not node.is_terminal():
			node = node.expand()

		# Simulation: Does not actually creat MCT Nodes 
		result = node.rollout()

		# Backpropagation: Update the root node 
		node.backpropagate(result, color)

	return root.best_child(c=1).action  # Return best move


def play_game():
	board = [['.']*9 for _ in range(9)]
	current_player = 1

	for turn in range(81):

		if current_player == 1:
			move = mcts_search(board, 1, iterations=80)
			print(f"MCTS plays: {move}")
		else:
			move = mcts_search(board, 2, iterations=80)
			print(f"MCTS plays: {move}")

		board[move[0]][move[1]] = current_player
		if current_player == 1: 
			if MCTSNode(board,1,...,(move[0], move[1])).check_winner():
				for row in board: 
					for ele in row: 
						if ele == 1: 
							print('●', end=' ')
						if ele == 2: 
							print('○', end=' ')
						if ele == '.':
							print('.', end=' ')
					print()
				print(f"BITCH {current_player} WINS!")
				return
		else: 
			if MCTSNode(board,2,...,(move[0], move[1])).check_winner():
				for row in board: 
					for ele in row: 
						if ele == 1: 
							print('●', end=' ')
						if ele == 2: 
							print('○', end=' ')
						if ele == '.':
							print('.', end=' ')
					print()
				print(f"BITCH {current_player} WINS!")
				return

		current_player = 1 if current_player == 2 else 2
	for row in board: 
		for ele in row: 
			if ele == 1: 
				print('●', end=' ')
			if ele == 2: 
				print('○', end=' ')
			if ele == '.':
				print('.', end=' ')
		print()
	print("Draw!")

play_game()