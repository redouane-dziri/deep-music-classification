from fetch_data import fetch_data_local
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, Dense
from tensorflow.keras import Sequential

'''
Template script for loading and training on the GLCMs

The network only works for mel_maps and spectrograms
'''

train_df = fetch_data_local(map_type='mel_map', train=True, angle='0')
test_df = fetch_data_local(map_type='mel_map', train=False, angle='0')

#Generate the set of labels
label_names = set([genre for genre in train_df['genre']])

#Encode the labels into ints
label_to_idx = dict((name, index) for index, name in enumerate(label_names))


#(Inelegant way of) regenerating the np.array from the maps
samples = []
labels = []
for indx, row in train_df.iterrows():
    samples.append(np.array(row['maps']))
    labels.append(label_to_idx[row['genre']])


BATCH_SIZE = 32

#Generating the tf.dataset for train
dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
dataset = dataset.shuffle(128).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#Set up the neural net
model = Sequential()
model.add(Conv2D(12,(6,6),activation='tanh',input_shape=(15,15,1)))
model.add(AveragePooling2D())
model.add(Conv2D(6,(3,3),activation='tanh'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#extracting the first batch from the train data
sample_batch, label_batch = next(iter(dataset))

#train_on_batch to check that it can overfit the batch
for i in range(1000):
    metrics = model.train_on_batch(tf.reshape(sample_batch,[BATCH_SIZE,15,15,1]), label_batch)
    if(i % 50==0):
        print("Loss: {}, Accuracy: {}".format(metrics[0], metrics[1]))