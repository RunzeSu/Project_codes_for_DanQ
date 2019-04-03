import numpy as np
from keras import optimizers
import h5py
import scipy.io
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc
np.random.seed(1337) # for reproducibility

from tensorflow import set_random_seed
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional



print ('loading data')
trainmat = scipy.io.loadmat('data/train.mat')
validmat = scipy.io.loadmat('data/valid.mat')
testmat = scipy.io.loadmat('data/test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(0,2,1))
y_train = np.array(trainmat['traindata'])

forward_lstm = LSTM(units=320, return_sequences=True)
backward_lstm = LSTM(units=320, return_sequences=True)
brnn1 = Bidirectional(forward_lstm)
brnn2 = Bidirectional(backward_lstm)


print ('building model')

model = Sequential()

model.add(Convolution1D(
                        filters=320,
			                  input_shape=(1000, 4),
                        kernel_size=26,
                        padding="valid",
                        activation="relu",
                        strides=1))

model.add(MaxPooling1D(pool_size=13, strides=13))

model.add(Dropout(0.2))

model.add(brnn1)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, units=919))
model.add(Activation('sigmoid'))

print ('compiling model')

rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=rms)

print ('running at most 50 epochs')

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit(X_train, y_train, batch_size=50, epochs=50, shuffle=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper])
#model.fit(X_train, y_train, batch_size=50, epochs=1, shuffle=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']))
print ('loading test data')

validmat = scipy.io.loadmat('data/valid.mat')
x = np.transpose(validmat['validxdata'],axes=(0,2,1))
#testmat.close()


print ('predicting on valid sequences')
y = model.predict(x, verbose=1)

pred=y
y_test = validmat['validdata']

for i in range(1,919):
    print(roc_auc_score( y_test[:,i], pred[:,i]))
    
print ('predicting on test sequences')
testmat = scipy.io.loadmat('data/test.mat')
x = np.transpose(testmat['testxdata'],axes=(0,2,1))
y = model.predict(x, verbose=1)
pred=y
y_test = testmat['testdata']

for i in range(1,919):
    print(roc_auc_score( y_test[:,i], pred[:,i]))

