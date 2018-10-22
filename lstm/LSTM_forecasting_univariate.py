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
MODELO PERSISTENCE FORECAST: La observación del time step t-1 se utiliza para predecir la observación en t. Se implementa
tomando la úlitma observación del training y del acumulado histórico del walk-forward validation, usando esto para 
predecir el momento t. Es muy sencillo así que lo usamos de Benchmark de performance.

WALK-FORWARD VALIDATION: La idea es por cada time step del dataset, iremos uno por uno. El modelo realizará una 
predicción del time step, luego, el verdadero valor esperado del test se tomará y estará disponible en el modelo para 
el siguiente time step. Esto básicamente se basa en el mundo real, donde cada nueva observación del precio estará
disponible cada día y será usado para el día siguiente. Finalmente, todas las predicciones en el test se usarán 
para calcular el RMSE (el RMSE permite penalizar errores altos y permite entender el valor en que están expresados
los datos)
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
df = df[['PESPANIA']]

# We split train-test (with temporal logic)
train, test = df.values[0: int(len(df)*0.70)], df.values[int(len(df)*0.70):]

# Walk-forward Validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    '''
    1-Historia tiene el training t-2, t-1... hasta t
    1- En predictions toma el último valor de la historia (t)
    2- Se agrega el primer valor del test (t+1) a la historia
    3-En predictions se toma el último valor de la historia (t+1)
    '''
    predictions.append(history[-1]) # Tomamos el último valor de la historia
    history.append(test[i]) # Acumulamos el valor real de cada test

rmse = np.sqrt(mean_squared_error(test, predictions))  # Esto es básicamente predecir t, con t-1
print('RMSE PERSISTENCE %.2f' % rmse)

plot.plot(test, label='test')
plot.plot(predictions, label='predict')
plot.show()

# EMPIEZA EL LSTM ################################################################################################

# WE CREATE A LAG
look_back = 1
for i in range(1, look_back + 1, 1):
    df['L' + str(i) + '_PESPANIA'] = df['PESPANIA'].shift(i)

# We reorder the columns
cols = df.columns.values.tolist()
df = df[[x for x in cols if x.startswith('L')] + ['PESPANIA']]


# TRANSFORM THE SERIE TO STATIONARY
'''
Si la serie es muy dependiente del tiempo, podemos remover la tendencia de las observaciones, y luego, agregarla 
nuevamente al forecast para que tenga la escala original y puede calcularse un RMSE comparable.
Una forma es diferenciar los datos, esto es en cada observacion hacer x(t) - x(t-1).
'''
raw_values = df.values
df['PESPANIA_diff'] = df['PESPANIA'].diff()
df = df.dropna()

# We normalize
df = df[['PESPANIA_diff']]
scaler = MinMaxScaler(feature_range=(-1, 1)) # Lo ponemos (-1, 1) porque la activation function es tanh, que espera valores (-1, 1)
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=['PESPANIA_diff'])

# We split train-test (with temporal logic)
train_x, test_x = df.values[0: int(len(df)*0.70), :-1], df.values[int(len(df)*0.70):, :-1]
train_y, test_y = df.values[0: int(len(df)*0.70), -1], df.values[int(len(df)*0.70):, -1]

# We need the data as [samples, time_steps, features] - Now is [samples, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

'''
We are going to use the memory between batches model so we have to define batch_input_shape. Este es un tuple
que espera como arguemtnos el número de observaciones a leer en cada batch, el número de time steps, y el número de
features. Define qué tan rápido se aprende de los datos (numero de epochs).
'''
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=True))
model.add(Dense(1))
'''
La red requiere como output una única neuran con una activación lineal para predecir el precio en el siguiente
time step
'''
early_stopping = EarlyStopping(patience=2)
model.compile(loss='mse', optimizer='adam')

for i in range(20):
    model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, callbacks=[early_stopping], shuffle=False)
    model.reset_states() # Debemos controlar cuando el state se resetea

# WALK-FORWARD VALIDATION LSTM
train_predict = model.predict(train_x, batch_size=batch_size)
train_predict = scaler.inverse_transform(train_predict)

predictions = list()
for i in range(len(test_x)):
    x, y = test_x[i], test_y[i]
    x = x.reshape(1, 1, len(test_x))
    y_hat = model.predict(x, batch_size=batch_size)
    y_hat = scaler.inverse_transform(y_hat)
    predictions.append(y_hat)
    expected = raw_values[len(train_y) + i + 1]
    print('time step, predicted, expected', (i+1, y_hat, expected))





