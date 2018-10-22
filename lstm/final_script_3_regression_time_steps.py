import pandas as pd
import STRING
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error, recall_score, precision_score, fbeta_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal
import seaborn as sns



np.random.seed(42)
os.chdir(STRING.root)

train_normal = pd.read_csv('train_normal.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
valid_normal = pd.read_csv('valid_normal.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
valid_mixed = pd.read_csv('valid_mixed.csv', sep=';', encoding='latin1', parse_dates=['DIA'])
test = pd.read_csv('test.csv', sep=';', encoding='latin1', parse_dates=['DIA'])

train_normal['CONTROL'] = pd.Series(0, index=train_normal.index)
valid_normal['CONTROL'] = pd.Series(1, index=valid_normal.index)
valid_mixed['CONTROL'] = pd.Series(2, index=valid_mixed.index)
test['CONTROL'] = pd.Series(3, index=test.index)

df = pd.concat([train_normal, valid_normal, valid_mixed, test], axis=0)
df = df[['DIA', 'PESPANIA', 'TARGET', 'CONTROL']]
df = df.sort_values(by=['DIA'], ascending=[True]).reset_index(drop=True)

cols = df.columns.values.tolist()

# df = df[cols[0:9] + ['PESPANIA', 'TARGET']]
df = df.set_index('DIA')


# WE CREATE A LAG
look_back = 7
for col in df.columns.values.tolist():
    if col.startswith('PESPANIA'):
        for i in range(1, look_back + 1, 1):
            df['L' + str(i) + '_' + col] = df[col].shift(i)


# LO PRIMERO ES ORDENAR PARA QUE QUEDEN LOS LAGS DE UN LADO Y LA VARIABLE Y DEL OTRO
cols = df.columns.values.tolist()
df = df[['PESPANIA'] + [x for x in cols if x.startswith('L') and 'PESPANIA' in x] + [x for x in cols if x.startswith('L') and
                                                                      not 'PESPANIA' in x] +
        ['CONTROL', 'TARGET']]
print(df)
# Stationarity
# df['PESPANIA'] = df['PESPANIA'].diff()
df = df.dropna()

# WE REINDEX (because of NAN)
df = df.reset_index(drop=False)
train_normal = df[df['CONTROL'] == 0]
valid_normal['CONTROL'] = df[df['CONTROL'] == 1]
valid_mixed['CONTROL'] = df[df['CONTROL'] == 2]
test['CONTROL'] = df[df['CONTROL'] == 3]
df = df.set_index('DIA')

# We normalize
df_c = df[['CONTROL', 'TARGET']]
df_c = df_c.reset_index(drop=False)
df = df.reset_index(drop=True)
df = df.drop(['CONTROL', 'TARGET'], axis=1)
cols = df.columns.values.tolist()
scaler = MinMaxScaler(feature_range=(-1, 1))
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=cols)
features_name = [x for x in cols if x.startswith('L')]
features = int(len(features_name) / look_back)

df = pd.concat([df, df_c], axis=1)
train_normal = df[df['CONTROL'] == 0]
valid_normal = df[df['CONTROL'] == 1]
valid_mixed = df[df['CONTROL'] == 2]
test = df[df['CONTROL'] == 3]
train_normal = train_normal.drop(['CONTROL', 'DIA', 'TARGET'], axis=1)
valid_normal = valid_normal.drop(['CONTROL', 'DIA', 'TARGET'], axis=1)
valid_mixed = valid_mixed.drop(['CONTROL', 'DIA', 'TARGET'], axis=1)
test = test.drop(['CONTROL', 'DIA', 'TARGET'], axis=1)

# LO SEGUNDO ES DEFINIR COMO "X" TODOS LOS LAGS y COMO "Y" AL PESPANIA
print(train_normal.columns.values.tolist())
print(train_normal)
train_normal_x, train_normal_y = train_normal.values[:, 1:], train_normal.values[:, 0]
valid_normal_x, valid_normal_y = valid_normal.values[:, 1:], valid_normal.values[:, 0]
valid_mixed_x, valid_mixed_y = valid_mixed.values[:, 1:], valid_mixed.values[:, 0]
test_x, test_y = test.values[:, 1:], test.values[:, 0]


# We need the data as [samples, time_steps, features] - Now is [samples, features]
train_normal_x = np.reshape(train_normal_x, (train_normal_x.shape[0], look_back, features)) # TERCERO: TENEMOS QUE DEFINIR SAMPLES, TIME_STEPS=LAGS, FEATURES=FEATURES
valid_normal_x = np.reshape(valid_normal_x, (valid_normal_x.shape[0], look_back, features))
valid_mixed_x = np.reshape(valid_mixed_x, (valid_mixed_x.shape[0], look_back, features))
test_x = np.reshape(test_x, (test_x.shape[0], look_back, features))

print(train_normal_x.shape, train_normal_y.shape, valid_normal_x.shape, valid_normal_y.shape)

# LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(train_normal_x.shape[1], train_normal_x.shape[2]), return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='adam')
print("Inputs: " + str(model.input_shape))
print("Outputs: " + str(model.output_shape))
print("Actual input: " + str(train_normal_x.shape))
print("Actual output:" + str(train_normal_y.shape))
early_stopping = EarlyStopping(patience=2)
history = model.fit(train_normal_x, train_normal_y, epochs=200, batch_size=50, validation_data=(valid_normal_x,
                                                                                               valid_normal_y),
                    verbose=2, shuffle=False, callbacks=[early_stopping])

# PLOT
plot.plot(history.history['loss'], label='train')
plot.plot(history.history['val_loss'], label='test')
plot.legend()
plot.show()

# PREDICTIONS
cols.remove('PESPANIA')
# Train
yhat = model.predict(train_normal_x)
train_normal_x = train_normal_x.reshape(train_normal_x.shape[0], look_back * features)
inv_yhat = np.concatenate((yhat, train_normal_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
train_prediction = inv_yhat[:, 0]


train_normal_y = train_normal_y.reshape(len(train_normal_y), 1)
inv_y = np.concatenate((train_normal_y, train_normal_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
train_true = inv_y[:, 0]

# Valid Mixed
yhat = model.predict(valid_mixed_x)
valid_mixed_x = valid_mixed_x.reshape(valid_mixed_x.shape[0], look_back * features)
inv_yhat = np.concatenate((yhat, valid_mixed_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
valid2_prediction = inv_yhat[:, 0]


valid_mixed_y = valid_mixed_y.reshape(len(valid_mixed_y), 1)
inv_y = np.concatenate((valid_mixed_y, valid_mixed_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
valid2_true = inv_y[:, 0]

# Test
yhat = model.predict(test_x)
test_x = test_x.reshape(test_x.shape[0], look_back * features)
inv_yhat = np.concatenate((yhat, test_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
test_prediction = inv_yhat[:, 0]


test_y = test_y.reshape(len(test_y), 1)
inv_y = np.concatenate((test_y, test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
test_true = inv_y[:, 0]

# PLOT
rsme = np.sqrt(np.power(train_true - train_prediction, 2))
mae = np.abs(train_true - train_prediction)
plot.figure(figsize=(15, 5))
plot.plot(train_true, label='true')
plot.plot(train_prediction, label='predicted')
plot.plot(mae, label='rmse')
plot.legend()
plot.title('Training Using 3 Timesteps')
plot.show()

# ERROR VECTOR TRAIN
train_error_vector = train_true - train_prediction
mean = np.mean(train_error_vector, axis=0)
cov = np.cov(train_error_vector, rowvar=False)
p_values = multivariate_normal.logpdf(train_error_vector, mean, cov)
plot.figure(figsize=(15, 5))
plot.plot(p_values)
plot.title('p-values')
plot.show()

# ERROR VECTOR OPTIMIZING THRESHOLD (VALID_2)
v2_error_vector = valid2_true - valid2_prediction
v2_p_values = multivariate_normal.logpdf(v2_error_vector, mean, cov)

predicted_valid = pd.DataFrame(valid2_prediction, columns=['predicted'])
v2_pvalues_df = pd.DataFrame(v2_p_values, columns=['LogPD'])
valid_mixed_c = df_c[df_c['CONTROL'] == 2].reset_index(drop=True)
test_c = df_c[df_c['CONTROL'] == 3].reset_index(drop=True)
valid_mixed_df = pd.concat([valid_mixed_c[['CONTROL', 'TARGET']], predicted_valid, v2_pvalues_df], axis=1)
valid_mixed_df['TARGET'] = valid_mixed_df['TARGET'].map(int)

thresholds = np.linspace(-50.0, 0.0, 1000)

scores = []

for threshold in thresholds:
    y_hat = [1 if e < threshold else 0 for e in valid_mixed_df.LogPD.values]
    y_hat = list(map(int, y_hat))

    scores.append([
        recall_score(y_pred=y_hat, y_true= valid_mixed_df.TARGET.values),
        precision_score(y_pred=y_hat, y_true=valid_mixed_df.TARGET.values),
        fbeta_score(y_pred=y_hat, y_true=valid_mixed_df.TARGET.values,
                    beta=1)])

scores = np.array(scores)
threshold = thresholds[scores[:, 2].argmax()]
print('final Threshold ', threshold)
predicted = [1 if e < threshold else 0 for e in valid_mixed_df.LogPD.values]
predicted = list(map(int, predicted))

print('PRECISION ', precision_score(valid_mixed_df.TARGET.values, predicted))
print('RECALL ', recall_score(valid_mixed_df.TARGET.values, predicted))
print('FBSCORE ', fbeta_score(valid_mixed_df.TARGET.values, predicted, beta=1))

# ERROR VECTOR TEST
test_error_vector = test_true - test_prediction
test_p_values = multivariate_normal.logpdf(test_error_vector, mean, cov)

# PLOTS
# Plot v2 Pvalues
plot.figure()
plot.plot(v2_p_values, label='log PD', color=sns.xkcd_rgb["dark teal"])
plot.axhline(y=threshold, ls='dashed', label='Threshold', color=sns.xkcd_rgb["dark teal"])
plot.legend(bbox_to_anchor=(1, .45), borderaxespad=0.)
plot.xlabel("Time step")
plot.ylabel("Log PD")
plot.title("Validation2 p-values")
plot.show()

# FINAL PLOT VALIDATION 2
f = plot.figure(figsize=(20, 10))
plot.subplots_adjust(hspace=0.1)
v2_below_threshold = np.where(v2_p_values <= threshold)
print(v2_below_threshold)
ax1 = plot.subplot(211)
ax1.plot(valid2_true, label='True Value', color=sns.xkcd_rgb["denim blue"])
ax1.plot(valid2_prediction, ls='dashed', label='Predicted Value', color=sns.xkcd_rgb["medium green"])
ax1.plot(abs(valid2_true - valid2_prediction), label='Error', color=sns.xkcd_rgb["pale red"])
for column in v2_below_threshold[0]:
    ax1.axvline(x=column, color=sns.xkcd_rgb["dark peach"], alpha=.5)

ax1.axvline(x=v2_below_threshold[0][-1], color=sns.xkcd_rgb["dark peach"], alpha=.5)
# for row in v2_true_anomalies:
#    plot.plot(row, validation2_true[row], 'r.', markersize=20.0)
ax1.legend(bbox_to_anchor=(1.02, .3), borderaxespad=0., frameon=True)
ax1.set_xticklabels([])
plot.ylabel("Power Price")
plot.title("Validation2. Using 3 timestep")

# plot v2 log PD
ax2 = plot.subplot(212)
values_date = np.arange(0, len(valid_mixed_c.index), 50)
valid_mixed_c['DIA'] = valid_mixed_c['DIA'].astype(str)
df_dates_xticks = valid_mixed_c[valid_mixed_c.index.isin(values_date)]
ax2.plot(v2_p_values, label='Log PD', color=sns.xkcd_rgb["dark teal"])
ax2.axhline(y=threshold, ls='dashed', label='Threshold', color=sns.xkcd_rgb["dark teal"])
ax2.legend(bbox_to_anchor=(1, .3), borderaxespad=0., frameon=True)
plot.ylabel("Log PD")
ax2.set_xticklabels(df_dates_xticks['DIA'], rotation=30, fontsize=8)
plot.xticks(values_date)
plot.title("Validation2 p-values")

# Set up the xlabel and xtick
# xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
plot.xlabel("Time step")
plot.show()

print(v2_below_threshold[0])
anomaly_dates_valid = valid_mixed_c[valid_mixed_c.index.isin(v2_below_threshold[0])]
print(anomaly_dates_valid)

# TEST EVALUATION
test_below_threshold = np.where(test_p_values <= threshold)
f = plot.figure(figsize=(20, 10))
plot.subplots_adjust(hspace=0.01)

ax1 = plot.subplot(211)
ax1.plot(test_true, label='True Value', color=sns.xkcd_rgb["denim blue"])
ax1.plot(test_prediction, ls='dashed', label='Predicted Value', color=sns.xkcd_rgb["medium green"])
ax1.plot(abs(test_true - test_prediction), label='Error', color=sns.xkcd_rgb["pale red"])
for column in test_below_threshold[0]:
    ax1.axvline(x=column, color=sns.xkcd_rgb["dark peach"], alpha=0.5)
# for row in test_true_anomalies:
#    plot.plot(row, test_true[row], 'r.', markersize=20.0)
ax1.legend(bbox_to_anchor=(1, 1), borderaxespad=0., frameon=True)
plot.ylabel("Power Price")
# plot.title("Test. Using 1 timestep")

ax2 = plot.subplot(212)
values_date = np.arange(0, len(test_c.index), 50)
test_c['DIA'] = test_c['DIA'].astype(str)
df_dates_xticks = test_c[test_c.index.isin(values_date)]
ax2.plot(test_p_values, label='Log PD', color=sns.xkcd_rgb["dark teal"])
ax2.axhline(y=threshold, ls='dashed', label='Threshold', color=sns.xkcd_rgb["dark teal"])
ax2.legend(bbox_to_anchor=(1, .3), borderaxespad=0., frameon=True)
plot.ylabel("Log PD")
ax2.set_xticklabels(df_dates_xticks['DIA'], rotation=30, fontsize=8)
plot.xticks(values_date)
plot.title("test p-values")

# Set up the xlabel and xtick
# xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
plot.xlabel("Time step")
plot.show()


print(test_below_threshold[0])
anomaly_dates_test = test_c[test_c.index.isin(test_below_threshold[0])]
print(anomaly_dates_test)
