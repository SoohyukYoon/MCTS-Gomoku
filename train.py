from mcts import MCTSNode, mcts_search
import pickle
import threading
# Global Flag
stop_training = False

# Saves the Game Tree after training
def save_game_tree(GameTree, filename='game_tree.pkl'):
	with open(filename, 'wb') as f: 
		pickle.dump(GameTree, f)
	print(f"GameTree saved to {filename}")

# Check for if user wants to end training
def input_listener():
	global stop_training
	while not stop_training: 
		user_input = input()
		if user_input.strip().lower() == "end training": 
			stop_training = True
			print("\n Training will pause after the current game completes ...")

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

def train():
	global stop_training
	stop_training = False 

	count_win_1 = 0
	count_win_2 = 0 
	draw_count = 0 
	games = 1000
	
	empty_board = ['.'] * 49 
	black = 1

	filename = input("Please enter trained model's filename: ")
	try: 
		if filename == '':
			GameTree = load_game_tree('game_tree.pkl')
			print(GameTree.visits)
		elif filename == 'empty': 
			GameTree = MCTSNode(empty_board, black)
		else: 
			GameTree = load_game_tree(filename)
	except ValueError:
		print("Invalid file or File does not exist, please try again")

	# Start input listener
	listener_thread = threading.Thread(target=input_listener, daemon=True)
	listener_thread.start()
	print("Type 'end training' at any time to stop training gracefully.\n")

	for game in range(games):
		if stop_training:
			print(f"\n Training stopped by user at game {game}")
			save_game_tree(GameTree)
			return

		print("___________________")
		print("GAME: ", game)
		draw = True
		current_player = 1

		board = ['.'] * 49
		black = 1
		root = MCTSNode(board, black)
		child = root
		for turn in range(49):
			print()
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
				loss_1 = True
				child = mcts_search(child, 1, loss_1, iterations=10000)
			else:
				loss_2 = True
				child = mcts_search(child, 2, loss_2, iterations=10000)

			board[child.action] = current_player
			if current_player == 1: 
				if MCTSNode(board,1,..., child.action).check_winner():
					print(f"BITCH {current_player} WINS!")
					# return 
					count_win_1 += 1 
					draw = False
					GameTree.add_game(root, 1)
					break
			else: 
				if MCTSNode(board,2,...,child.action).check_winner():
					print(f"BITCH {current_player} WINS!")
					# return
					count_win_2 += 1 
					draw = False
					GameTree.add_game(root, 2)
					break
			current_player = 1 if current_player == 2 else 2

		if draw: 
			print("Draw!")
			draw_count += 1
			GameTree.add_game(root, 0)
		print("Current win count: ")
		print("1: ", count_win_1)
		print("2: ", count_win_2)
		print("Draws: ", draw_count)
		for i, ele in enumerate(board):  
			if ele == 1: 
				print('●', end=' ')
			if ele == 2: 
				print('○', end=' ')
			if ele == '.':
				print('.', end=' ')
			if i % 7 == 6: 
				print()
	
	save_game_tree(GameTree)

train()
