# Checked looks about good 

from cnn import GameDataset, organize_games

# Generate GameDataset
organize_games('renjunet_v10_20180803.xml')

# Generate numpy dataset objects
train_dataset = GameDataset(root='dataset', split='training')
valid_dataset = GameDataset(root='dataset', split='validation')

# Print the board states
state_b, state_w, state_e, action_t = train_dataset.__getitem__(0)
print("State T, b: ", state_b)
print("State T, w: ", state_w)
print("State T, e: ", state_e)
print("Action T: ", action_t)

state_b_v, state_w_v, state_e_v, action_t_v = train_dataset.__getitem__(0)
print("State V, b: ", state_b_v)
print("State V, w: ", state_w_v)
print("State V, e: ", state_e_v)
print("Action V: ", action_t_v)


