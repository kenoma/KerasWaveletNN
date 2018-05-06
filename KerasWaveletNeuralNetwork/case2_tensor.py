
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

x = Amplitude_line1 - sum(Amplitude_line1) / len(Amplitude_line1)
x = x / max(x)

y = Marking_line1
X = []
Y = []
slice = 100
for i in range(slice, len(x)):
    if((y[i] != 0) or (i % 100 == 1)):
        X.append([a for a in x[(i - slice) : i]])
        tmpy = np.zeros(3)
        tmpy[int(round(y[i]))] = 1
        Y.append(tmpy)

print("Total samples:",len(X))

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
x_train = X[1:len(X) - 100]
y_train = Y[1:len(X) - 100]
x_test = X[len(X) - 100:len(X)]
y_test = Y[len(X) - 100:len(X)]

wavelons = 100

f_layer = WNN(wavelons, 20,
                     input_dim = slice)

model = Sequential()
model.add(f_layer)
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['categorical_accuracy'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs = 100,
          verbose = 1,
          batch_size = 100)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)

pred = model.predict_classes(np.array(x_test))
px = []
py = []
for i in range(0,len(y_test)):
    if(pred[i] != 0):
        px.append(i)
        py.append(x_test[i][-1])

print(px)

plt.ion()
plt.show()
plt.clf()
plt.title('tensor sig_1')
plt.ylabel('n')
plt.xlabel('y')

plt.plot([a[-1] for a in x_test], label='test', c=(0,0,0))
plt.scatter(px, py, c=(1,0,0), alpha=0.5, s=10)
plt.show()
plt.pause(120)