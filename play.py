from mcts import MCTSNode, mcts_search
import pickle

# Loads the Game Tree for testing
def load_game_tree(filename='game_tree.pkl'):
	try:
		with open(filename, 'rb') as f: 
			GameTree = pickle.load(f)
		print(f"GameTree loaded from {filename}")
		return GameTree
	except FileNotFoundError: 
		print(f"File {filename} was not found, please try a valid filename")
		return None

def play():

	empty_board = ['.'] * 49 
	black = 1
	GameTree = MCTSNode(empty_board, 1)

	filename = input("Please enter trained model's filename: ")
	try: 
		if filename == '':
			GameTree = load_game_tree('game_tree.pkl')
		else: 
			GameTree = load_game_tree(filename.strip())
	except ValueError:
		print("Invalid file or File does not exist, please try again")
	
	print("___________________")
	print("GAME: ")
	draw = True
	current_player = 1
	loss = True

	board = ['.'] * 49
	black = 1
	child = GameTree
	for turn in range(49):
		print()
		move = 0
		for i, ele in enumerate(board):  
			if ele == 1: 
				print('●', end=' ')
			if ele == 2: 
				print('○', end=' ')
			if ele == '.':
				print('.', end=' ')
			if i % 7 == 6: 
				print()
		
		if current_player == 1:
			next_child = child.best_child()
			if next_child: 
				child = next_child 
				move = child.action
			else: 
				print("No child found")
				temp_node = MCTSNode(board[:], 1, parent=child, action=None)
				child = mcts_search(temp_node, 1, loss, iterations=10000)
			move = child.action
			board[move] = current_player 
		else:
			move_str = input("Please enter your move: ")
			try: 
				row_str, col_str = move_str.split(',')
				move = int(row_str.strip()) * 7 +  int(col_str.strip())

				board[move] = current_player
				
				found = False
				for c in child.children: 
					if c.action == move: 
						child = c
						found = True
						break 
				if not found: 
					child = MCTSNode(board[:], 2, parent=child,action=move)
			except ValueError: 
				print("Bad input value")


		if current_player == 1: 
			if MCTSNode(board,1,..., child.action).check_winner():
				print(f"BITCH {current_player} WINS!")
				# return 
				draw = False
				break
		else: 
			if MCTSNode(board,2,...,child.action).check_winner():
				print(f"BITCH {current_player} WINS!")
				# return
				draw = False
				break
		current_player = 1 if current_player == 2 else 2

	for i, ele in enumerate(board):  
		if ele == 1: 
			print('●', end=' ')
		if ele == 2: 
			print('○', end=' ')
		if ele == '.':
			print('.', end=' ')
		if i % 7 == 6: 
			print()

play()