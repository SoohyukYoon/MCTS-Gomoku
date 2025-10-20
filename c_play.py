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

# Black is AI
state_b = torch.tensor([0]*225, dtype=torch.float32)
state_w = torch.tensor([0]*225, dtype=torch.float32)
state_e = torch.tensor([1]*225, dtype=torch.float32)
state   = torch.stack([state_b, state_w, state_e]).reshape(1, 3, 15, 15).to(device)

# Create Board to print
board = [['.'] * 15 for _ in range(15)]

# Iterate through max 225 moves 
for i in range(225): 
	# Print board 
	for j in range(15): 
		print(j + 1, end='  ') if j < 9 else print(j + 1, end=' ')
		for k in range(15): 
			print(board[j][k], end=' ')		
		print() 
	print('   a b c d e f g h i j k l m n o')

	# If black's move make move
	if i % 2 == 1:
		with torch.no_grad(): 
			print("hewo")
			output = model(state)
			output = output[0].tolist()
			while True: 
				move = output.index(max(output))
				row = move // 15 
				col = move % 15
				output[move] = 0 
				# print(output)
				# print(f"AI trying position: row={row}, col={col}")
				# print(f"White occupied: {state[0, 1, row, col]}, Black occupied: {state[0, 0, row, col]}")
				if state[0, 1, row, col] != 1 and state[0, 0, row, col] != 1:
					break 
			state[0, 0, row, col] = 1
			state[0, 2, row, col] = 0
			board[row][col] = '○'

	# If white's move make move
	if i % 2 == 0: 
		while True:
			move_str = input("Please enter your move: ")
			try: 
				row_str, col_str = move_str.split(',')
				row = int(row_str.strip()) - 1
				col = ord(col_str.strip()) - ord('a') 
				if state[0, 1, row, col] == 1 or state[0, 0, row, col] == 1:
					print("Stone already present.")
					continue 
				state[0, 1, row, col] = 1
				state[0, 2, row, col] = 0
				board[row][col] = '●'
				break
			except ValueError: 
				print("Bad input value")