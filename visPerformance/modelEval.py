from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

df = pd.read_csv('raster_cuda_final.csv')

##### From Matt's paper
# Raterization Model -> O + (VO*PPT)
# O -> objects
# VO -> min(AP, O)
# AP -> 55% * 1/numMPiTasks^1/3 * pixels
# PPT -> AP * 4

cores = df['Cores']
objects = df['objects']
pixels = df['pixels']
actualPixels = df ['actual pixels']

activePixels = .55 * (1/cores**(1/3)) * pixels
#visibleObjects = min(activePixels, objects)
#cheat on this for now
visibleObjects = activePixels
ppt = activePixels*4
secondParam = ppt*visibleObjects
secondParam.name = "secondParam"

#tempX = [objects, secondParam, actualPixels, cores]
tempX = [objects, ppt, actualPixels, cores]
X =  pd.DataFrame(tempX)
X = X.T # transpose tow/columns


mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

#X = df[['pixels', 'objects', 'Cores', 'actual pixels']]
y = df['Total Render']

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, shuffle=True)


early_stop = EarlyStopping(monitor='mean_squared_error', patience=20, verbose=1)

for x in range(1, 64):
    for y in range(1, 64):
        for z in range(1, 64):
            # create model
            model = Sequential()
            model.add(Dense(x, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(y, activation='relu'))
            model.add(Dense(z, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'mae', 'mape', 'cosine'])
            history = model.fit(X_train, y_train, epochs=4, verbose=0, callbacks=[early_stop])

            currentPrediction = model.predict(X_test)
            currentPrediction = currentPrediction.reshape(currentPrediction.size)
            currentPrediction.shape
            # Explained variance score: 1 is perfect prediction
            print('Params[%i,%i,%i] -- R2 Variance Score: %.2f' % (x, y, z, r2_score(y_test, currentPrediction)))
