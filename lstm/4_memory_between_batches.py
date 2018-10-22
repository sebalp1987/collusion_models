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

"""
LSTM es capaz de recordar a lo largo de secuencias largas. Normalmente, el estado de la red se resetea despues
de cada batch termina de entrenar (epochs). Podemos ganar mayor control determinando cuándo la red se resetea haciendo
que el layer del LSTM sea "stateful". Esto implica que podemos construir estados sobre todo el training e incluso
mantenter este estado si es necesario.
Requiere que los datos no esten shuffle. Esto es porque queremos que aprenda de la secuencia de las observaciones y no 
que haga un shuffle dentro del epoch.
Requiere explícitamente resetear la red despues de cada epoch (models.reset_states()). Esto implica que tenemos
que crear nuestro propio loop fuera de los epochs y dentro de los epochs.
"""


np.random.seed(42)
os.chdir(STRING.root)

train_normal = pd.read_csv('train_normal.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
valid_normal = pd.read_csv('valid_normal.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
valid_mixed = pd.read_csv('valid_mixed.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
test = pd.read_csv('test.csv', sep=';', encoding='latin1', parse_dates=['DIA'])

train_normal = train_normal[['DIA', 'PESPANIA', 'TARGET']]
valid_normal = valid_normal[['DIA', 'PESPANIA', 'TARGET']]
valid_mixed = valid_mixed[['DIA', 'PESPANIA', 'TARGET']]
test = test[['DIA', 'PESPANIA', 'TARGET']]

df = pd.concat([train_normal, valid_normal, valid_mixed, test], axis=0)
df = df.sort_values(by=['DIA'], ascending=[True]).reset_index(drop=True)

# We normalize
df = df[['PESPANIA']]
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=['PESPANIA'])

# WE CREATE A LAG
look_back = 3  # It is the number of previous time steps to use as input to predict the next time period
for i in range(1, look_back + 1, 1):
    df['L' + str(i) + '_PESPANIA'] = df['PESPANIA'].shift(i)

# We reorder the columns
cols = df.columns.values.tolist()
df = df[[x for x in cols if x.startswith('L')] + ['PESPANIA']]
df = df.dropna()

# We split train-test (with temporal logic)
train_x, test_x = df.values[0: int(len(df)*0.70), :-1], df.values[int(len(df)*0.70):, :-1]
train_y, test_y = df.values[0: int(len(df)*0.70), -1], df.values[int(len(df)*0.70):, -1]

# We need the data as [samples, time_steps, features] - Now is [samples, features]
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

# We fit the model
model = Sequential()
# Lo primero es definir el stateful = True, y en lugar de especificar el input dimension, tenemos que harcodear
# el número de samples en cada batch (batch_size), time_steps, featueres usando ahora BATCH_INPUT_SHAPE
batch_size = 10
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
early_stopping = EarlyStopping(patience=2)
model.compile(loss='mse', optimizer='adam')

# Aquí creamos nuestro propio loop con epochs = 20
for i in range(20):
    model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, callbacks=[early_stopping], shuffle=False) # Importante definir el shuffle = False
    model.reset_states()

# Predictions
train_predict = model.predict(train_x, batch_size=batch_size) # El predict se hace también por batches
model.reset_states()
test_predict = model.predict(test_x)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])

test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# MSE
mse_train = mean_squared_error(train_y[0], train_predict[:, 0])
mse_test = mean_squared_error(test_y[0], test_predict[:, 0])
print('MSE Train %.2f' % np.sqrt(mse_train))
print('MSE Test %.2f' % np.sqrt(mse_test))
# Mejora un poquito respecto al window


# PLOT
true_vales = scaler.inverse_transform(df[['PESPANIA']])
true_vales = pd.DataFrame(true_vales, columns=['true'])

# As we are predicting t+1 we have to shift the predictions + 1
train_predict = pd.DataFrame(train_predict, columns=['train_predict'])
train_predict.index += look_back

test_predict = pd.DataFrame(test_predict, columns=['test_predict'])
test_predict.index += len(train_predict.index) # We shift the test index to continue the train

plot.plot(true_vales, label='true_values')
plot.plot(train_predict, label='predict_train')
plot.plot(test_predict, label='predict_test')
plot.legend()
plot.show()