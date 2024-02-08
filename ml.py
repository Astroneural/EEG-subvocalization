import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.layers import Conv1D, Conv2D, Dropout, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dense, Dropout,  MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, LSTM
import keras_tuner as kt
import data_pca

import matplotlib.pyplot as plt

x_train = data_pca.x_train_ica
y_train = data_pca.y_train
x_test = data_pca.x_test_ica
y_test = data_pca.y_test

model = Sequential()

model.add(Conv1D(200, kernel_size=225, activation='relu', input_shape=(129,5), padding='same'))

model.add(layers.MaxPooling1D(pool_size=2))

model.add(GlobalAveragePooling1D())

model.add(Dropout(0.15))

#model.add(Dense(1028, activation='relu'))
model.add(Dense(3, activation='softmax')) 

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
              metrics=['accuracy'])

checkpoint_filepath = '/Users/shaum/eeg-stuffs/checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history = model.fit(x_train, y_train, batch_size=1024, epochs=100, shuffle=True, validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback])

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
print(f'best accuracy: {max(val_acc_per_epoch)}')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()