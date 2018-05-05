import keras
from wnn_layer import WNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt

# Generate dummy data
import numpy as np

x = []
y = []
x_test = []
y_test = []
x_n = 0.01
l = 3.8

for i in range(0,10000):
    x_old = x_n
    x_n = l * x_n * (1 - x_n)
    x_nplus = l * x_n * (1 - x_n)
    x.append([x_old, x_n])
    y.append([x_nplus])

for i in range(0, 100):
    x_old = x_n
    x_n = l * x_n * (1 - x_n)
    x_nplus = l * x_n * (1 - x_n)

    x_test.append([x_old, x_n])
    y_test.append([x_nplus])

    
x_train = np.array(x)
y_train = np.array(y)

model = Sequential()
f_layer = WNN(1000, 1, input_dim=2)
model.add(f_layer)

model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['mae'])

model.fit(x_train, y_train,
          epochs=200,
          verbose=1,
          batch_size=100)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)
pred = model.predict(np.array(x_test))

plt.ion()
plt.show()
plt.clf()
plt.title('Logistics map')
plt.ylabel('x[n-1]')
plt.xlabel('x[n]')

plt.plot(y_test, label='test', c=(0,0,0))
plt.plot(pred, label='pred', c=(1,0,0))
#plt.scatter([a[0] for a in x_train], [a[0] for a in y_train], c=(0,0,0), alpha=0.5,s=1)
#plt.scatter([a[0] for a in x_test], [a[0] for a in pred], c=(1,0,0), alpha=0.5,s=1)
plt.show()
plt.pause(120)

