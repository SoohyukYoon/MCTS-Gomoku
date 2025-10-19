import matplotlib.pyplot as plt

def plot_train_loss(history): 
	# fig: entire canvas 
	# ax: actual plot where data is drawn 
	fig, ax = plt.subplots(figsize=(6,4))
	ax.plot(hishistory = {
			'train_loss': [],
			'val_loss': []
		}tory.get('all_loss', []), label='train_loss')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Loss')
	ax.set_title('Training Curve')
	ax.legend()
	fig.tight_layout() # shows the labels I've defined
	return fig, ax


history = train(___populate___)
fig, ax = plot_train_loss(history)
plt.show()