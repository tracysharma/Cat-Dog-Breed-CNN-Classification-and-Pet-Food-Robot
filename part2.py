import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
import pickle
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.inception_v3 import preprocess_input
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics
import pydot

# Loading files
# To test with other files (for example, with the background removed) this file names should be changed to the appropriate ones and everything works fine
X = pickle.load(open("X.pickle","rb"))
X = np.array(X)

y = pickle.load(open("y.pickle","rb"))
y = np.array(y)

# Split data into training and testing data
# This split is stratified, which means that the ratios between the numbers of images in each class will be kept equal in the testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Function that returns the (complex) model according to some variables
def getModel(dropout=.25, learning_rate=0.001, augmentation=False):
  base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
  base_model.trainable = False

  inputs = tf.keras.Input(shape=(299, 299, 3))
  if augmentation:
    x = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])(inputs)
  else:
      x = inputs
  x = tf.keras.applications.inception_v3.preprocess_input(x)
  x = base_model(x, training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(256,activation='relu')(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  outputs = tf.keras.layers.Dense(37,activation='softmax')(x)
  model = tf.keras.Model(inputs, outputs)
  model.summary()

  model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

  return model

# Function to encode y to arrays of 0's and 1's so it checks out with the model we have
onehot_encoder = OneHotEncoder(sparse=False)
def onehotencode_func(y):
  integer_encoded = y.reshape(len(y), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  return onehot_encoded

# Defining values that will be experimented with and setting up k-fold cross validation
learning_rate_list = [0.001]
dropout_values_list = [0.25]

# 3-fold cross validation will be used because its computationally easier/faster
kfold = StratifiedKFold(n_splits=3, shuffle=True)

# dictionary where data will be stored
hist = {'learning_rate': {}, 'neurons': {}, 'dropout': {}}

# Functions used later to save data from hyper parameter tuning on files, and to read them as well
def save_in_file(parameter, dict, filename):
  f = open(filename, 'ab')
  pickle.dump({parameter : dict[parameter]}, f)
  f.close()

# returns a list
def read_file(filename):
  objs = [] 
  f = open(filename, 'rb')
  while 1:
      try:
          objs.append(pickle.load(f))
      except EOFError:
          break
  f.close()
  return objs

# Function where information from k-fold cross validation will be averaged, stored and returned
# the 'model_func' parameter is a lambda function
def test_params(lr, model_func):
  dic = {}
  i = 0.0

  # splitting data into the folds
  folds = kfold.split(x_train, y_train)
  for train_index, val_index in folds:

    # getting the model with the desired parameters
    model = model_func(lr)

    x_train_kf, x_val_kf =  x_train[train_index], x_train[val_index]
    y_train_kf, y_val_kf = onehotencode_func(y_train[train_index]), onehotencode_func(y_train[val_index])

    # training the model with data from the train data folds
    historytemp = model.fit(x_train_kf, y_train_kf, batch_size=32, epochs=5, validation_data=(x_val_kf, y_val_kf))

    del model

    if dic == {}:
      # if dictionary is empty, values will be put there
      dic['train_acc'] = np.array(historytemp.history['accuracy'])
      dic['train_loss'] = np.array(historytemp.history['loss'])
      dic['val_acc'] = np.array(historytemp.history['val_accuracy'])
      dic['val_loss'] = np.array(historytemp.history['val_loss'])
    else:
      # if dictionary is not empty, values will be added element wise
      dic['train_acc'] += np.array(historytemp.history['accuracy'])
      dic['train_loss'] += np.array(historytemp.history['loss'])
      dic['val_acc'] += np.array(historytemp.history['val_accuracy'])
      dic['val_loss'] += np.array(historytemp.history['val_loss'])
    
    i+=1

  for k in dic:
    # each number in each array in the dictionary will be divided by the number of iterations, producing the mean of all the values read
    dic[k] /= i

  return dic

# Setting up the experiences
# To use the model with the data augmentation layers, a parameter augmentation=True should be passed to the getModel functions

# To use the simpler model, the function that the lambda functions are calling should be changed to getSimplerModel
# changing learning rate:
# lr_model_func = lambda x : getModel(learning_rate=x)
# for lr in learning_rate_list:
#   hist['learning_rate'][lr] = test_params(lr, lr_model_func)

# save_in_file('learning_rate', hist, 'data.pickle')


# # chaning dropout value:
# drop_model_func = lambda x : getModel(dropout=x)
# for d in dropout_values_list:
#   hist['dropout'][d] = test_params(d, drop_model_func)

# save_in_file('dropout', hist, 'data.pickle')


# print(read_file('data.pickle'))

# # Displaying a representation of the neural network architecture
# # Just like before, adding the parameter augmentation=True to the getModel function will add the data augmentation layers
# # first we get a model
# m = getModel()

# # then we generate the diagram
# diagram_file = 'model_diagram_complex.png'
# im = tf.keras.utils.plot_model(
#     m, to_file=diagram_file, show_shapes=False, show_dtype=False,
#     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
# )

# # this displays the diagram without title
# # display(im)

# # to add a title:
# img = [cv2.imread(diagram_file)]

# fig = plt.figure(figsize=(15,15))
# grid = ImageGrid(
#     fig,
#     111,
#     nrows_ncols=(1,1),
#     axes_pad=0.7
# )

# for ax, im in zip(grid, img):
#     ax.imshow(im)
#     ax.axis('off')

# fig.subplots_adjust(top=.95)
# fig.suptitle('Complex Neural Network Architecture Diagram', size='xx-large')
# plt.show()



# Displaying the confusion matrix and metrics table
# This was done after analysing the data
# learning rate = 0.001
# dropout value = 0.25
# First we get a new model and train it with all the training data available:

model = getModel(learning_rate=0.001, dropout=0.25)
model.fit(x_train, onehotencode_func(y_train), batch_size=32, epochs=5)

# Getting the predictions of the test data and transforming it to a number (selecting the index of the maximum value and summing one):
y_pred = model.predict(x_test)
y_pred2 = [ np.argmax(i)+1 for i in y_pred]


# list of labels in order gotten from the previous notebook
labels = ['abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua', 'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'maine_coon', 'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian_blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese', 'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

# this function creates the confusion matrix
# the first argument is the expected results, and the second is the obtained previsions
cm = metrics.confusion_matrix(y_test, y_pred2)

fig, ax = plt.subplots( figsize= (10, 10))

ax.imshow(cm)

ax.set_xticks(range(0, 37))
ax.set_yticks(range(0,37))

ax.set_yticklabels(labels)
ax.set_xticklabels(labels, rotation='vertical')

plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix', size='xx-large')
plt.show()

# this next function produces a table with values of precision, recall, f1-score and accuracy
metrics.classification_report(y_test, y_pred2)

# Fine-tuning transfer learning model
# Having a model already trained on our data, we are going to unfreeze the previously frozen layers
for l in model.layers:
    if l.name == 'inception_v3':
        base_model = l
        break

# unfreeze
base_model.trainable = True

# check how many layers in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# And we are going to freeze all the layers before one specific chosen layer (rougly at 2/3 of the layers)
# Fine-tune from this layer onwards
fine_tune_at = 208

# Freeze all the layers before the fine_tune_at layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

# The learning rate used in this process should be lower because the model to be trained is huge, so smaller steps are better in order for it not to overfit Now we fit on the data once again for 5 more epochs
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.RMSprop(lr=0.001/10),
              metrics=['accuracy'])

fine_tune_epochs = 5
total_epochs =  5 + fine_tune_epochs
model.fit(x_train, onehotencode_func(y_train), batch_size=32,initial_epoch=5, epochs=total_epochs, validation_split=0.3)