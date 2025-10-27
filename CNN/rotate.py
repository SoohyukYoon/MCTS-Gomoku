import numpy as np 
move = 92
moves = np.full(225, '.')

print("Zero Degrees:")
moves[move] = '●'
print(moves.reshape(15, 15))
moves.flatten()
moves[move] = '.'

print("90 Degrees")
move = (move // 15) + (14 - (move % 15)) * 15
moves[move] = '●'
print(moves.reshape(15, 15))
moves.flatten()
moves[move] = '.'

print("180 Degrees")
move = (move // 15) + (14 - (move % 15)) * 15
moves[move] = '●'
print(moves.reshape(15, 15))
moves.flatten()
moves[move] = '.'

print("270 Degrees")
move = (move // 15) + (14 - (move % 15)) * 15
moves[move] = '●'
print(moves.reshape(15, 15))
moves.flatten()
moves[move] = '.'