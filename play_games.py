from supervised_cnn import *
from unsupervised_cnn import * 
from selfplay_mcts import *
from pathlib import Path

device = "cpu"
print(f"Using device: {device}")

# Check for winner in Gomoku 
def check_winner(state, player, action): 
	dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
	for dir in dirs: 
		winningCells = [[action // 15, action % 15]]
		for sgn in (1, -1): 
			rowStep = dir[0]
			colStep = dir[1]
			curRow = (action // 15) + rowStep * sgn
			# self.action[0] + rowStep * sgn
			curCol = (action % 15) + colStep * sgn
			# self.action[1] + colStep * sgn
			# print("Coords: ", curRow, curCol)
			if (curRow >= 15 or curRow < 0) or (curCol >= 15 or curCol < 0):
				continue
			while state[curRow * 15 + curCol] == player: 
				winningCells.append([curRow, curCol])
				if len(winningCells) > 5: 
					return None
				curRow += rowStep * sgn
				curCol += colStep * sgn
				if (curRow >= 15 or curRow < 0) or (curCol >= 15 or curCol < 0): 
					break 
		# print(len(winningCells))
		if len(winningCells) == 5: 
			print(f"WINNER IS: {player} at ({action // 15}, {action % 15})")
			return True
	return False

# Win and Draw counts: 
transform_win = 0 
non_transform_win = 0 
draw_count = 0 

# Game was a won 
won = False
# Create Board to print
board = [['.'] * 15 for _ in range(15)]
# Initialize Game
game = ["h8","h9","j9","g8","j10","g7","g10","i10","j11","j12","f7","i11","i12","k10","g9","f9","i6","k13","l14","j7","i7","j6","g6","j4","j5","i5","h6","h4","k7","i4","k4","h7","f4","g3","f2","g5","f6","e6","l5","m5","m6","n7","k5","k3","k8","k6","l7","n5","l3","m2","l4","l6","l2","l1","e10","f10","e8","d9","e9","e11","f5","f8","f12","k11","k12","d12","d11","l10","m9","e13","f14","f13","g13","e14","g12","g11","e12","h12","i13","b10","c12","c6","d5","c5","c7","b6","e3","b8","b7","d6","a6","b12","b11","d10","c9","n10","n9","l12","m11","o10","m10"]

# Iterate through max 225 moves 
for i in range(225): 
	# If black's move make move
	if i % 2 == 0:
		print(f"played at: {game[i]}")
		move = game[i] 
		row = int(move[1:]) - 1
		col = ord(move[0]) - ord('a') 
		board[row][col] = 1
		# check if game ended: 
		action = row * 15 + col
		board_flat = [ele for row in board for ele in row]
		black = 1
		if check_winner(board_flat, black, action): 
			non_transform_win += 1 
			won = True
			break
		
	# If white's move make move
	if i % 2 == 1: 
		print(f"played at: {game[i]}")
		move = game[i] 
		row = int(move[1:]) - 1
		col = ord(move[0]) - ord('a') 
		board[row][col] = 2
		# check if game ended: 
		action = row * 15 + col
		board_flat = [ele for row in board for ele in row]
		white = 2
		if check_winner(board_flat, white, action): 
			transform_win += 1 
			won = True
			break
	

	# Print board 
	for j in range(15): 
		print(j + 1, end='  ') if j < 9 else print(j + 1, end=' ')
		for k in range(15): 
			if board[j][k] == 1: 
				print('●', end=' ')
			elif board[j][k] == 2: 
				print('○', end=' ')
			else: 
				print('.', end=' ')	
		print() 
	print('   a b c d e f g h i j k l m n o')
	print()
	
if not won: 
	draw_count += 1

print("Game: ", g)
print("Non transform win count: ", non_transform_win)
print("Transform win count: ", transform_win)

