
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

x = Amplitude_line1-sum(Amplitude_line1)/len(Amplitude_line1)
x = x / max(x)
X = []
Y = []
slice = 10
for i in range(slice, len(x)-1):
    X.append([a for a in x[(i - slice):i]])
    Y.append(x[i+1])

print("Total samples:",len(X))

x_train = X[1:len(X)-10000]
y_train = Y[1:len(X)-10000]
x_test = X[len(X)-10000:len(X)]
y_test = Y[len(X)-10000:len(X)]

wavelons = 100
f_layer = WNN(wavelons, 1, input_dim = slice)

model = Sequential()
model.add(f_layer)

model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['mae'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs = 200,
          verbose = 1,
          batch_size = 100)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)

pred = model.predict(np.array(x_test))

plt.ion()
plt.show()
plt.clf()
plt.title('tensor sig 1')
plt.ylabel('n')
plt.xlabel('y')

plt.plot(y_test, label='test', c=(0,0,0))
plt.plot(pred, label='pred', c=(1,0,0))
plt.show()
plt.pause(120)