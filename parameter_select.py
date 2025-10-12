from mcts import MCTSNode, mcts_search

def parameter_tournament():
	
	# Control parameters
	control = {
		'win' : 1,
		'draw' : 0,
		'loss' : -1,
	}

	# Confident parameters --- Even if win ratio low still does it


	
	count_win_1 = 0
	count_win_2 = 0 
	draw_count = 0 
	games = 1000
	
	empty_board = ['.'] * 49 
	black = 1

	for game in range(games):
		print("___________________")
		print("GAME: ", game)
		draw = True
		current_player = 1

		board = ['.'] * 49
		black = 1
		root = MCTSNode(board, black)
		child = root
		for turn in range(49):
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
					break
			else: 
				if MCTSNode(board,2,...,child.action).check_winner():
					print(f"BITCH {current_player} WINS!")
					# return
					count_win_2 += 1 
					draw = False
					break
			current_player = 1 if current_player == 2 else 2

		if draw: 
			print("Draw!")
			draw_count += 1
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

parameter_tournament()
