import pickle
import numpy as np

import matplotlib.pyplot as plt

# Function to read pickle files to a list
def read_file(f):
  objs = [] 
  f = open(f, 'rb')
  while 1:
      try:
          objs.append(pickle.load(f))
      except EOFError:
          break
  f.close()
  return objs

# Reading the files with historic information about the hyper parameter tuning
# The name of the file passed to the read_file function should be changed to the correct name of the file
l = read_file('data_v2.pickle')
lr = l[-2]['learning_rate']
d = l[-1]['dropout']

# dictionary that maps the keys of the dictionaries above to the "normal" name of the metric
metric2name = {'train_acc' : 'train accuracy', 'train_loss' : 'train loss', 'val_acc': 'validation accuracy', 'val_loss': 'validation loss'}

# Displaying charts
# Evaluating different learning rates:
fig, axs = plt.subplots(2, figsize=(10,7))

i = 0
for val in lr:
    for metric in lr[val]:
        axs[i].plot(lr[val][metric], label=metric2name[metric])
    axs[i].set_title('learning rate = ' + str(val))
    axs[i].set_xlabel('epochs')
    axs[i].set_ylabel('value')
    axs[i].legend(loc='upper right')
    i +=1

fig.suptitle('Evolution of metrics varying learning rate', size='xx-large')
fig.tight_layout()
plt.show()

# Evaluating different dropout values:
fig, axs = plt.subplots(2, figsize=(10,7))

fig.suptitle('Evolution of metrics varying dropout value', size='xx-large')
i = 0
for val in d:
    for metric in d[val]:
        axs[i].plot(d[val][metric], label=metric2name[metric])
    axs[i].set_title('dropout value = ' + str(val))
    axs[i].set_xlabel('epochs')
    axs[i].set_ylabel('value')
    axs[i].legend(loc='upper right')
    i +=1
    
fig.tight_layout()
plt.show()

# to compare the two charts:

d[0.25]['train_loss']- d[0.35]['train_loss']
#d[0.25]['train_acc']- d[0.35]['train_acc']
d[0.25]['val_acc']- d[0.35]['val_acc']
d[0.25]['val_loss']- d[0.35]['val_loss']

# to compare the two charts:

lr[0.001]['train_loss']- lr[0.01]['train_loss']
# lr[0.001]['train_acc']-  lr[0.01]['train_acc']
# lr[0.001]['val_acc']-    lr[0.01]['val_acc']
# lr[0.001]['val_loss']- lr[0.01]['val_loss']
