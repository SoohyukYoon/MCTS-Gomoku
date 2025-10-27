from a_cnn import CNN, GameDataset
import torch  
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import transforms as T

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Model initialized!")

# Load the trained weights
model = CNN().to(device)
print("Model initialized!")
# weights_only: security, if you use internet models might have malicious code, so only extracts the weights
model.load_state_dict(torch.load('model_weights.pth', weights_only=True, map_location=torch.device('cpu'))) 
# Set to eval mode to play 
model.eval()

# Load the trained weights
model_t = CNN().to(device)
print("Model initialized!")
# weights_only: security, if you use internet models might have malicious code, so only extracts the weights
model_t.load_state_dict(torch.load('model_weights_transform.pth', weights_only=True, map_location=torch.device('cpu'))) 
# Set to eval mode to play 
model_t.eval()


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
for g in range(1): 	
	# Game was a won 
	won = False

	# Black is AI
	state_b = torch.tensor([0]*225, dtype=torch.float32)
	state_w = torch.tensor([0]*225, dtype=torch.float32)
	state_e = torch.tensor([1]*225, dtype=torch.float32)
	state   = torch.stack([state_b, state_w, state_e]).reshape(1, 3, 15, 15).to(device)

	# Create Board to print
	board = [['.'] * 15 for _ in range(15)]

	# Iterate through max 225 moves 
	for i in range(225): 
		# If black's move make move
		if i % 2 == 0:
			if i == 0: 
				state[0, 0, 7, 7] = 1
				state[0, 2, 7, 7] = 0
				board[7][7] = 1
				continue
			with torch.no_grad(): 
				output = model(state)
				output = output[0].tolist()

				while True: 
					move = output.index(max(output))
					# print(output[move])
					row = move // 15 
					col = move % 15
					output[move] = -float("inf")

					# So on first oupt, 0nly 0,0 for some reason has loss 0  
					# print(output)
					# print(f"AI trying position: row={row}, col={col}")
					# print(f"White occupied: {state[0, 1, row, col]}, Black occupied: {state[0, 0, row, col]}")
					if state[0, 1, row, col] != 1 and state[0, 0, row, col] != 1:
						break 
				state[0, 0, row, col] = 1
				state[0, 2, row, col] = 0
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
			with torch.no_grad(): 
				output = model(state)
				output = output[0].tolist()
				while True: 
					move = output.index(max(output))
					row = move // 15 
					col = move % 15
					output[move] = -float("inf")
					# So on first oupt, 0nly 0,0 for some reason has loss 0  
					#print(output)
					# print(f"AI trying position: row={row}, col={col}")
					# print(f"White occupied: {state[0, 1, row, col]}, Black occupied: {state[0, 0, row, col]}")
					if state[0, 1, row, col] != 1 and state[0, 0, row, col] != 1:
						break 
				state[0, 1, row, col] = 1
				state[0, 2, row, col] = 0
				board[row][col] = 2
			# while True:
			# 	move_str = input("Please enter your move: ")
			# 	try: 
			# 		row_str, col_str = move_str.split(',')
			# 		row = int(row_str.strip()) - 1
			# 		col = ord(col_str.strip()) - ord('a') 
			# 		if state[0, 1, row, col] == 1 or state[0, 0, row, col] == 1:
			# 			print("Stone already present.")
			# 			continue 
			# 		state[0, 1, row, col] = 2
			# 		state[0, 2, row, col] = 0
			# 		board[row][col] = 2
			# 		break
			# 	except ValueError: 
			# 		print("Bad input value")
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

