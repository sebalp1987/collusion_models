import STRING
import pandas as pd
from fbprophet import Prophet
import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plot
import seaborn as sns

df = pd.read_csv(STRING.final_file, sep=';', dtype={'FECHA': str, 'ANIO': str, 'MES': str, 'DIA': str})

del df['HORA']
del df['FECHA_HORA']

df.index.name = None
df.reset_index(inplace=True)

df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')
df['ANIO'] = df['ANIO'].map(int)
df['MES'] = df['MES'].map(int)
df['DIA'] = df['DIA'].map(int)


df = df.groupby(['FECHA']).mean().reset_index()


df['ds'] = df['FECHA']
df.set_index(['ds'], inplace=True)
df.index.name = None

auction_date = df.FECHA[df['DUMMY'] == 1].tolist()


days_before = 30

for auction_date in auction_date:

    start_date =auction_date - datetime.timedelta(days=days_before)

    anormalY = df[df['FECHA'] >= auction_date - datetime.timedelta(days=days_before)]
    anormalY = anormalY[anormalY['FECHA'] <= auction_date]
    anormalY = anormalY[['PESPANIA']]

    df_before_auction = df[df['FECHA'] < auction_date - datetime.timedelta(days=days_before)]
    df_before_auction = df_before_auction[['FECHA', 'PESPANIA']]
    df_before_auction.columns = ['ds', 'y']
    
    normalY = df[df['FECHA'] >= auction_date]
    normalY = normalY[normalY['FECHA'] <=auction_date + datetime.timedelta(days=days_before)]
    normalY = normalY[['PESPANIA']]

    df_before_auction1 = df[df['FECHA'] <=auction_date]
    df_before_auction1 = df_before_auction1[['FECHA', 'PESPANIA']]
    df_before_auction1.columns = ['ds', 'y']

    fileModel = Prophet(interval_width=0.95)
    fileModel2 = Prophet(interval_width=0.95)

    fileModel.fit(df_before_auction)
    fileModel2.fit(df_before_auction1)

    # Predecimos los siguientes días para el anormal
    future_dates = fileModel.make_future_dataframe(periods=30, freq='D')
    y_hat = fileModel.predict(future_dates)
    y_hat = y_hat['yhat']
    prediction_anormal = y_hat[-31:].reset_index(drop=True)
    anormalY = anormalY.reset_index(drop=True)

    # Predecimos los siguientes días para el normal
    future_dates = fileModel2.make_future_dataframe(periods=30, freq='D')
    y_hat = fileModel2.predict(future_dates)
    y_hat = y_hat['yhat']
    prediction_normal = y_hat[-31:].reset_index(drop=True)
    y_test = normalY.reset_index(drop=True)

    print('MSE ANORMAL ', mean_squared_error(anormalY, prediction_anormal))
    print('R2 ANORMAL ', r2_score(anormalY, prediction_anormal))


    #Comparamos con el verdadero valor

    print('MSE NORMAL ', mean_squared_error(y_test, prediction_normal))
    print('R2 NORMAL ', r2_score(y_test, prediction_normal))

    prediction_normal = pd.DataFrame(prediction_normal, index=y_test.index)
    prediction_normal = pd.concat([y_test, prediction_normal], axis=1)
    prediction_normal.columns = ['PESPANIA_REAL_NO_COLUSION', 'PESPANIA_PRED_NO_COLUSION']
    prediction_normal['DIF_PORC'] = (prediction_normal['PESPANIA_REAL_NO_COLUSION'] - prediction_normal[
                'PESPANIA_PRED_NO_COLUSION']) / prediction_normal['PESPANIA_PRED_NO_COLUSION']
    print('PRECIO PROMEDIO PREDICHO - NO COLUSION %.5f' % prediction_normal['PESPANIA_PRED_NO_COLUSION'].mean())
    print('PRECIO PROMEDIO REAL - NO COLUSION %.5f ' % prediction_normal['PESPANIA_REAL_NO_COLUSION'].mean())
    print('DIFERENCIA PROMEDIO PORCENTUAL (REAL/PRED -1)', prediction_normal['DIF_PORC'].mean() * 100, '%')

    prediction_anormal = pd.DataFrame(prediction_anormal, index=anormalY.index)
    prediction_anormal = pd.concat([anormalY, prediction_anormal], axis=1)
    prediction_anormal.columns = ['PESPANIA_REAL_COLUSION', 'PESPANIA_PRED_COLUSION']
    prediction_anormal['DIF_PORC'] = (prediction_anormal['PESPANIA_REAL_COLUSION'] - prediction_anormal[
                'PESPANIA_PRED_COLUSION']) / prediction_anormal['PESPANIA_PRED_COLUSION']
    print('PRECIO PROMEDIO PREDICHO - COLUSION %.5f' % prediction_anormal['PESPANIA_PRED_COLUSION'].mean())
    print('PRECIO PROMEDIO REAL - COLUSION %.5f' % prediction_anormal['PESPANIA_REAL_COLUSION'].mean())
    print('DIFERENCIA PROMEDIO PORCENTUAL (REAL/PRED -1)', prediction_anormal['DIF_PORC'].mean() * 100, '%')
    # prediction_anormal.to_csv('prediction_anormal_dia.csv', sep=';', index=False)



    fig, ax = plot.subplots()
    prediction_anormal = prediction_anormal.reset_index()
    sns.regplot(y='PESPANIA_PRED_COLUSION', x='index', data=prediction_anormal, ax=ax,
                        label='PREDICTED')
    sns.regplot(y='PESPANIA_REAL_COLUSION', x='index', data=prediction_anormal, ax=ax,
                        label='REAL')
    # diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label='perfect prediction')
    plot.legend(loc='best')
    plot.title('Differences between Prices')
    plot.show()

    fig, ax = plot.subplots()
    prediction_normal = prediction_normal.reset_index()
    sns.regplot(y='PESPANIA_PRED_NO_COLUSION', x='index', data=prediction_normal, ax=ax,
                        label='PREDICTED')
    sns.regplot(y='PESPANIA_REAL_NO_COLUSION', x='index', data=prediction_normal, ax=ax,
                        label='REAL')
    # diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label='perfect prediction')
    plot.legend(loc='best')
    plot.title('Differences between Prices')
    plot.show()





