import pandas as pd
import STRING
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import os


np.random.seed(42)
os.chdir(STRING.root)

train_normal = pd.read_csv('train_normal.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
valid_normal = pd.read_csv('valid_normal.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
valid_mixed = pd.read_csv('valid_mixed.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
test = pd.read_csv('test.csv', sep=';', encoding='latin1', parse_dates=['DIA'])


df = pd.concat([train_normal, valid_normal, valid_mixed, test], axis=0)
df = df.sort_values(by=['DIA'], ascending=[True]).reset_index(drop=True)
cols = df.columns.values.tolist()
df = df[cols[0:9] + ['PESPANIA']]
df = df.set_index('DIA')
del df['FECHA']

# WE CREATE A LAG
look_back = 1
for col in df.columns.values.tolist():
    for i in range(1, look_back + 1, 1):
        df['L' + str(i) + '_' + col] = df[col].shift(i)

# We reorder the columns
cols = df.columns.values.tolist()
df = df[['L1_PESPANIA'] + [x for x in cols if x.startswith('L') and 'PESPANIA' not in x] + ['PESPANIA']]

# Stationarity
# df = df.diff()
df = df.dropna()

# We normalize
cols = df.columns.values.tolist()
df = df.reset_index(drop=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
print(df)
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=cols)



# We split train-test (with temporal logic)
train_x, test_x = df.values[0: int(len(df)*0.70), :-1], df.values[int(len(df)*0.70):, :-1]
train_y, test_y = df.values[0: int(len(df)*0.70), -1], df.values[int(len(df)*0.70):, -1]
print(test_x.shape)
# We need the data as [samples, time_steps, features] - Now is [samples, features]
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1]) # Define al 1 como el timestep (ES la primera columna) y el resto como features
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
print("Inputs: " + str(model.input_shape))
print("Outputs: " + str(model.output_shape))
print("Actual input: " + str(train_x.shape))
print("Actual output:" + str(train_y.shape))
history = model.fit(train_x, train_y, epochs=10, batch_size=10, validation_data=(test_x, test_y), verbose=2, shuffle=False)

# PLOT
plot.plot(history.history['loss'], label='train')
plot.plot(history.history['val_loss'], label='test')
plot.legend()
plot.show()

# PREDICTIONS
yhat = model.predict(test_x)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])  # Transform [sample-timestep-features] a give shape [samples - features]
cols.remove('PESPANIA')
inv_yhat = np.concatenate((yhat, test_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]


test_y = test_y.reshape(len(test_y), 1)
inv_y = np.concatenate((test_y, test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
df = pd.DataFrame(inv_y, columns=['real'] + cols)
print(df)
inv_y = inv_y[:, 0]

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('RMSE % .2f' % rmse)

