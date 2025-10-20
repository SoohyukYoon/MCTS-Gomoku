from a_cnn import *
import torch  
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import transforms as T

# Initialize the Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#### Create Model --- No transform #####

# Create sample instance
transform = False
organize_games('renjunet_v10_20180803.xml', transform)

# Create DatasLoader instance
batch_size = 32 
train_dataset = GameDataset(root='dataset', split='training')
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle='True')

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
torch.save(model.state_dict(), 'model_weigths.pth')



#### Create Model --- With transform ####

# Create sample instance
transform = True
organize_games('renjunet_v10_20180803.xml', transform)

# Create DatasLoader instance
batch_size = 32 
train_dataset = GameDataset(root='dataset', split='training')
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle='True')

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

