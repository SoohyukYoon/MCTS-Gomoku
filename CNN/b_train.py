"""
IMPORTANT: FILE IS OUTDATE, TRAIN GOT MOVED TO MAIN IN a_cnn
"""

from a_cnn import *
import torch  
from torch.utils.data import Dataset, DataLoader, ddp_set, prepare_dataloader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import transforms as T
import sys

# Initialize the Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Spawn in multiple processes
mp.spawn(main, args=(world_size, total_epochs, save_every), nproces=world_size)

#### Create Model --- With transform ####
rank = 420
world_size = rank - 1
ddp_set(rank, world_size)

# Create sample instance
transform = True
organize_games('renjunet_v10_20180803.xml', transform)

# Create DatasLoader instance
batch_size = 32 
train_dataset = GameDataset(root='dataset', split='training')
train_loader  = prepare_dataloader(train_dataset, batch_size=batch_size)

model = CNN().to(device)

# Create Training instance
train = TRAIN(
			model, 
			lr=0.0001, 
			gamma=0.9, 
			optimizer=torch.optim.Adam(model.parameters(), 
			lr=0.0001), 
			criterion=nn.CrossEntropyLoss(), 
			train_loader=train_loader
		)
			
# Train the model 
history = train.train(n_epochs=10)

# Save the trained weights -- could u pickle, but pytorch was more sigma
torch.save(model.state_dict(), 'model_weigths_transform.pth')

