import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import pathlib
from keras import utils, regularizers

import matplotlib.pyplot as plt
import pandas as pd

from keras import layers
from keras import models
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.layers import Conv1D, Conv2D, Dropout, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dense, Dropout,  MaxPooling2D, GlobalAveragePooling2D
import keras_tuner as kt

from scipy import signal
from scipy.signal import butter, lfilter, medfilt, welch
from skimage.restoration import denoise_wavelet
from sklearn.decomposition import FastICA

def convert(csv):
  return genfromtxt(csv, delimiter=',')

train_data = convert('train_1-4.csv')
test_data = convert("preprocessing_data/1-23_3.csv")
val_data = convert('preprocessing_data/1-23_4.csv') # swapping these two lead to different results. 
# This way ~70% is achievable with val accuracies similar. Swapped test accuracy goes to ~75% but there is greater discrepancies.

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def convolve(data):
  filter = signal.firwin(400, [0.01, 0.06], pass_zero=False)
  return signal.convolve(data, filter, mode='same')

def medfit(data):
    med_filtered=signal.medfilt(data, kernel_size=3)
    return  med_filtered

def perform_fft(data):
    fft_data = np.fft.fft(data)
    return fft_data

def preprocess(data):
  nsamples = data[:, 1].shape[0]
  fs = 128.0
  lowcut = 0.1
  highcut = 60.0

  data = medfit(data)
  data = perform_fft(data)

  data[:, 2][1:] = butter_bandpass_filter(data[:, 2][1:], lowcut, highcut, fs, order=1)
  data[:, 3][1:] = butter_bandpass_filter(data[:, 3][1:], lowcut, highcut, fs, order=1)
  data[:, 4][1:] = butter_bandpass_filter(data[:, 4][1:], lowcut, highcut, fs, order=1)
  data[:, 5][1:] = butter_bandpass_filter(data[:, 5][1:], lowcut, highcut, fs, order=1)
  data[:, 6][1:] = butter_bandpass_filter(data[:, 5][1:], lowcut, highcut, fs, order=1)

  
  return data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_with_pca(data, n_components=5):
    nsamples = data.shape[0]
    fs = 128.0
    lowcut = 1
    highcut = 60.0

    # Apply Butterworth bandpass filter to each channel
    for i in range(2, data.shape[1]):
        data[:, i][1:] = butter_bandpass_filter(data[:, i][1:], lowcut, highcut, fs, order=1)

    # Apply median filter
    for i in range(2, data.shape[1]):
        data[:, i] = medfit(data[:, i])

    # Apply PCA
    pca_data = data[:, 2:]  # Exclude the first two columns (timestamps and labels)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)  # Standardize the data
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Concatenate the timestamps and labels with the PCA result
    preprocessed_data = np.hstack((data[:, :2], pca_result))

    return preprocessed_data

# Example usage:
preprocessed_train_data_with_pca = preprocess_with_pca(train_data)
preprocessed_test_data_with_pca = preprocess_with_pca(test_data)
preprocessed_val_data_with_pca = preprocess_with_pca(val_data)

preprocess(train_data)
preprocess(test_data)

preprocess(val_data)

n_chan = 5


def convert_to_image_array_with_pca(data, n_components, image_length):
    line_index = 1
    current_word = 0
    current_image = np.zeros((1, n_components))
    image_directory = []
    answer_directory = []

    while line_index < data.shape[0]:
        current_line = data[line_index]

        if int(current_line[0]) == current_word:
            current_image = np.vstack((current_image, current_line[2:])) # adding the data to the current image
        else: # if the word has changed, save this image and make a new one
            current_word = int(current_line[0])
            current_image_trimmed = current_image[1:image_length+1]  
            image_directory.append(current_image_trimmed)
            answer_directory.append(current_line[1])
            current_image = np.zeros((1, n_components))

        line_index += 1

    image_directory = np.array(image_directory) # shape (n_images, image_length, n_components)
    answer_directory = np.array(answer_directory) # shape (n_images,)
    answer_directory = utils.to_categorical(answer_directory)

    return image_directory, answer_directory

# Define parameters
n_components = 5  # Number of components after PCA
image_length = 128  # sampling rate and duration (128 Hz, 1s samples)

# Convert preprocessed data with PCA to image arrays
train_data_imageDirectory, traindata_answerDirectory = convert_to_image_array_with_pca(preprocessed_train_data_with_pca, n_components, image_length)
test_data_imageDirectory, test_data_answerDirectory = convert_to_image_array_with_pca(preprocessed_test_data_with_pca, n_components, image_length)
val_data_imageDirectory, val_data_answerDirectory = convert_to_image_array_with_pca(preprocessed_val_data_with_pca, n_components, image_length)

test_size = 0.2

x_train = train_data_imageDirectory
y_train = traindata_answerDirectory
x_test = test_data_imageDirectory
y_test = test_data_answerDirectory

np.random.seed(0)

def apply_ica(data, n_components):
    data = np.reshape(data, (data.shape[0], -1))  # Flatten channels and time_points
    ica = FastICA(n_components=n_components)
    ica_result = ica.fit_transform(data)

    return ica_result

n_components_ica = 5

ica_result_train = apply_ica(x_train, n_components_ica)
ica_result_test = apply_ica(x_test, n_components_ica)
ica_result_val = apply_ica(val_data_imageDirectory, n_components_ica)

ica_result_train_expanded = np.expand_dims(ica_result_train, axis=1)  # Expand dimensions along time_points axis
ica_result_test_expanded = np.expand_dims(ica_result_test, axis=1)
ica_result_val_expanded = np.expand_dims(ica_result_val, axis=1)

# Concatenate ICA features with original features
x_train_ica = np.concatenate((x_train, ica_result_train_expanded), axis=1)
x_test_ica = np.concatenate((x_test, ica_result_test_expanded), axis=1)
x_val_ica = np.concatenate((val_data_imageDirectory, ica_result_val_expanded), axis=1)

print("Shape of combined training data:", x_train_ica.shape)
print("Shape of combined testing data:", x_test_ica.shape)
print("Shape of combined validation data:", x_val_ica.shape)