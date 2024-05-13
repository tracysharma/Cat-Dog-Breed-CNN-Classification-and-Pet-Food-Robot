import os
import glob
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# First you need to download dataset from Kaggle and extract the zip file in this directory befrore starting

# Getting list of images (excluding some that do not open)
ROOT = 'archive'
IMGS_PATH = 'archive/images/images'
TRIMAP_PATH = 'archive/annotations/annotations/trimaps'

bad = {'Abyssinian_34.jpg', 'Egyptian_Mau_145.jpg', 'Egyptian_Mau_139.jpg', 'Egyptian_Mau_191.jpg', 'Egyptian_Mau_177.jpg', 'Egyptian_Mau_167.jpg'}

all_imgs = [i for i in os.listdir(IMGS_PATH) if i.rsplit('.',1)[1] == 'jpg' and i not in bad]
all_trimaps = [i for i in os.listdir(TRIMAP_PATH) if i.rsplit('.',1)[1] == 'png']

print('We have ' + str(len(all_imgs)) + ' good imgages.')
print('(for some reason, there are ' + str(len(bad)) + ' that do not want to open)')

# To get information about the images from the list.txt file
# Information is extracted into 2 dictionaries: info_by_id and info_by_breed
# getting info from 'list.txt' file
l = open(ROOT + '/annotations/annotations/list.txt', 'r')
get_breed = lambda pic : pic.rsplit('_',1)[0].lower()
get_species = lambda num : 'cat' if num==1 else 'dog'

info_by_id = {}
info_by_breed = {}

# taking note of the names and ids of the breeds
for line in l:
  if line[0] == '#':
    continue
  line = line.strip().split(' ')
  species = get_species(int(line[2]))
  id = int(line[1])
  breedid = int(line[3])
  name = get_breed(line[0]).lower()
  if name not in info_by_breed:
    info_by_breed[name] = {'breed' : name, 'species' : species, 'globalid': id, 'breedid':breedid, 'count':0}
    info_by_id[id] = info_by_breed[name]

# to count the images we can't trust the file
for p in [get_breed(n) for n in all_imgs]:
  info_by_breed[p]['count']+=1

# Display a bar chart of the number of images per breed & couting images per species
ids = list(info_by_id.keys())

# X value:
counts = [info_by_id[id]['count'] for id in ids]
x_labels = [info_by_id[i]['breed'] for i in ids]

# colours & legend:
colours = [ 'blue' if info_by_id[id]['species']=='cat' else 'red' for id in ids]

colours_leg = {'cat': 'blue', 'dog':'red'}
labels = list(colours_leg.keys())
handles = [plt.Rectangle((0,0),1,1, color=colours_leg[label]) for label in colours_leg]

# plotting:
fig, ax = plt.subplots( figsize= (11,4))
ax.bar(ids, counts, color=colours)

# set ticks & axis labels & legend:
ax.set_xticks(ids)
ax.set_xticklabels(x_labels, rotation='vertical')
plt.legend(handles, labels)
plt.xlabel('breeds')
plt.ylabel('no. of pictures')
plt.title('Image distribution by breed')
plt.show()

nr_cats = sum([ info_by_id[id]['count'] for id in ids if info_by_id[id]['species'] == 'cat' ])
nr_dogs = sum([ info_by_id[id]['count'] for id in ids if info_by_id[id]['species'] == 'dog' ])
print('There are ' + str(nr_cats) + ' images of cats, and ' + str(nr_dogs) + ' of dogs!')

# Reading all images, resizing and adding to list
# setting image pixel side size
IMG_SIZE = 299

def getXy(rem_background=False, imgs=None):
  # function that returns the number correspondent to the breed of the animal in the image, given the image name
  get_class_no = lambda name : info_by_breed[get_breed(name)]  ['globalid']
  
  # all image tensors will be stored here after resizing
  training_data = []
  
  for img in all_imgs:
    path = os.path.join(IMGS_PATH, img)
  
    # this is a trick to make the image be opened in RGB format, which is not the default
    img_array = cv2.imread(path)[...,::-1] 
  
    if rem_background:
      trimap_filename = img.rsplit('.', 1)[0] + '.png'
      if trimap_filename in all_trimaps:
        tri_array = cv2.imread(os.path.join(TRIMAP_PATH, trimap_filename))
        # if pixel in tri_array is 2, then it is background => 0
        tri_array[tri_array==2] = 0
        # if pixel in tri_array is < 2, then it is background and not defined => 1
        tri_array[tri_array>0] = 1
        
        # then it is multiplied so that the background pixels multiply by 0 and get "removed"
        img_array = np.multiply(tri_array, img_array)

    # here the images are rezise
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  
    # get the ID of the image class
    class_no = get_class_no(img)
  
    if imgs is not None and class_no not in imgs:
      imgs[class_no] = path
  
    training_data.append([img_array, class_no])
    
  # data should be in random order to improve performance
  random.shuffle(training_data)
  
  # separating data from list
  training = list(zip(*training_data))
  X = training[0]
  y = training[1]
  
  # transforming X to an np.array and resizing
  X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  return X, y

# Saving this data to files to make it easier to use it
def save(obj, fic_name, open_type='wb'):
  pickle_out = open(fic_name, open_type)
  pickle.dump(obj, pickle_out)
  pickle_out.close()

# a dictionary that is going to be used to map the ID to a path to an image
imgs = {}

X, y = getXy(imgs=imgs, rem_background=False)
save(X, 'X.pickle')
save(y, 'y.pickle')
print(X.shape)

# Save data from images without background as well
X, y = getXy(rem_background=True)
save(X, 'X_noBG.pickle')
save(y, 'y_noBG.pickle')
print(X.shape)
