#PCA

import pandas as pd

data = pd.read_csv('c_final.csv')
data = data.iloc[:,[1,2,3,4,5,6,7]]

# split data
features = ['i', 'v', 'freq', 'reactive_power','active_power', 'apparent_power']
x = data.loc[:, features].values
y = data.loc[:,'labels'].values

from sklearn.model_selection import train_test_split
train_fts, test_fts, train_lbl, test_lbl = train_test_split( x, y, test_size=0.15, random_state=0)

# standardise features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_fts) #only fit on training set (best practice)

train_fts = scaler.transform(train_fts)
test_fts = scaler.transform(test_fts)

# reduce dimensions from 6 --> 4
from sklearn.decomposition import PCA
pca = PCA(0.95) #preserves 95% variability

pca.fit(train_fts)

train_fts = pca.transform(train_fts)
test_fts = pca.transform(test_fts)

# reshape according to NN
train_fts = train_fts.reshape(train_fts.shape[0], train_fts.shape[1], 1)
test_fts = test_fts.reshape(test_fts.shape[0], test_fts.shape[1], 1)

# encode targets

import numpy as np

arr = np.unique(train_lbl)
i=0
train_targets = np.array([])

while (i<len(train_lbl)):
    val = train_lbl[i]
    
    if (val == arr[0]):
        map = 0
    if (val == arr[1]):
        map = 1
    if (val == arr[2]):
        map = 2
        
    train_targets = np.append(train_targets,map)
    i = i + 1

arr1 = arr
j=0
test_targets = np.array([])

while (j<len(test_lbl)):
    val1 = test_lbl[j]
    
    if (val1 == arr1[0]):
        map1 = 0
    if (val1 == arr1[1]):
        map1 = 1
    if (val1 == arr1[2]):
        map1 = 2
        
    test_targets = np.append(test_targets,map1)
    j = j + 1

import tensorflow
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique

model1 = Sequential()
model1.add(Conv1D(64, 2, activation="relu", input_shape=(4,1)))
model1.add(Dense(16, activation="relu"))
model1.add(MaxPooling1D())
model1.add(Flatten())
model1.add(Dense(3, activation = 'softmax'))
model1.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])

EPOCHS = 10
BATCH_SIZE = 32

from tensorflow.keras.callbacks import ModelCheckpoint
cp = ModelCheckpoint('model1/', save_best_only=True)

history = model1.fit(
    train_fts, train_targets, validation_data =
    (test_fts, test_targets), epochs=EPOCHS,
    batch_size=BATCH_SIZE, verbose=2, shuffle=True, callbacks=[cp])

predictions = model1.predict(test_fts, verbose='0')
for i in range(0, 20):
    print('Prediction: ', predictions[i],
          ', True value: ', test_targets[i])

pred = pd.DataFrame(predictions, columns = ['v1','v2','v3'])
    
# summation(softmax * actual value)
pred1 = pred.copy()

pred1['v1'] = pred1['v1']*arr[0]
pred1['v2'] = pred1['v2']*arr[1]
pred1['v3'] = pred1['v3']*arr[2]
pred1['final'] = pred1['v3'] + pred1['v2'] + pred1['v1']

val_test = pd.read_csv('c_final.csv')
val_test = val_test.iloc[8500:,[6]]

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

print("MAE")
print(mean_absolute_error(pred1['final'], val_test['apparent_power']))
print("MSE")
print(mean_squared_error(pred1['final'], val_test['apparent_power']))
print("MAPE")
print(mean_absolute_percentage_error(val_test['apparent_power'], pred1['final']))