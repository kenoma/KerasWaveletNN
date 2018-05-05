
import numpy as np
import keras
from wnn_layer import WNN
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random as rnd
from matplotlib.patches import Ellipse

Amplitude_line1, Marking_line1, Amplitude_line2, Marking_line2, Amplitude_line3, Marking_line3, Amplitude_line4, Marking_line4 = np.genfromtxt("train_5820_3333.csv", delimiter = ';', unpack = True,skip_header=1)

x = Amplitude_line1[:len(Amplitude_line1)-1]-Amplitude_line1[1:]
y = Marking_line1[1:]
X = []
Y = []
slice = 1
for i in range(slice, len(x) - slice):
    X.append([a for a in x[(i - slice):(i + slice)]])
    tmpy = np.zeros(3)
    tmpy[int(round(y[i]))] = 1
    Y.append(tmpy)

print("Total samples:",len(X))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

wavelons = 40

f_layer = WNN(wavelons, 100,
                     input_dim = 2 * slice)

model = Sequential()
model.add(f_layer)
model.add(Dense(25, activation='softmax'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['categorical_crossentropy'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs = 200,
          verbose = 1,
          batch_size = 10)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)
