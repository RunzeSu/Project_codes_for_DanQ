import sys
import numpy as np
import h5py
import scipy.io
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Bidirectional


forward_lstm = LSTM(input_shape=(3,), output_dim=320, return_sequences=True)
backward_lstm = LSTM(input_shape=(3,), output_dim=320, return_sequences=True)
brnn1 = Bidirectional(forward_lstm)
brnn2 = Bidirectional(backward_lstm)


print ('building model')

model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

model.add(MaxPooling1D(pool_length=13, stride=13))

model.add(Dropout(0.2))

model.add(brnn1)


model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, output_dim=919))
model.add(Activation('sigmoid'))

print ('compiling model')
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.load_weights('data/DanQ_bestmodel.hdf5')

print ('loading test data')

testmat = scipy.io.loadmat('data/test.mat')
x = np.transpose(testmat['testxdata'],axes=(0,2,1))
#testmat.close()

print ('predicting on test sequences')
y = model.predict(x, verbose=1)


print ('saving to aaa')
f = h5py.File('aaa', "w")
f.create_dataset("pred", data=y)
f.close()
