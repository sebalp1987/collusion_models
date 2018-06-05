import seaborn as sns
import pandas as pd
import STRING
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.dates as mdates
import matplotlib as mpl
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

sns.set(color_codes=True)

df = pd.read_csv(STRING.final_file, sep=';', dtype={'FECHA': str, 'ANIO': str, 'MES': str, 'DIA': str})


def extreme_random_byhour(df):
    # Log de Precios
    # df['PESPANIA'] = np.log(df['PESPANIA'])
    # df['PPORTUGAL'] = np.log(df['PPORTUGAL'])

    df['HORA'] = df['HORA'] - 1

    df = df[df['HORA'] != 24]

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

    df['DUMMY_2010_REGIMEN'] = pd.Series(0, index = df.index)
    df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1

    df['WEEKDAY'] = df['FECHA'].dt.dayofweek

    del df['DIA']
    del df['FECHA']
    del df['FECHA_HORA']
    del df['PPORTUGAL']

    # TARGET VARIABLE
    dummy_important = ['DUMMY', 'DUMMY_30_DAY', 'DUMMY_10_DAY', 'DUMMY_15_DAY', 'DUMMY_20_DAY']
    for i in dummy_important:
        del df[i]


    # statistics_basic()

    # DUMMIES
    dummy_var = ['ANIO', 'MES', 'HORA', 'WEEKDAY']
    for i in dummy_var:
        name = str(i)
        dummy = pd.get_dummies(df[i], prefix=name)
        df = pd.concat([df, dummy], axis=1)
        del dummy
        del df[i]


    # LAGS
    lag_number = 24*7
    lag_variables = ['PESPANIA', 'TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES',
                     'TOTAL_EXPORTACIONES_ES', 'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES',
                     'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                     'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL', 'PRICE_OIL',
                     'PRICE_GAS', 'RISK_PREMIUM']

    for i in range(1, lag_number, 1):
            for j in lag_variables:
                name = str(j) + '_lag_' + str(i)
                df[name] = df[j].shift(i)



    lag_number = 24*1
    climaticas = ['TME_MADRID', 'TMAX_MADRID', 'TMIN_MADRID', 'PP_MADRID', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN', 'PP_BCN'
                  ]
    for i in range(1, lag_number, 1):
            for j in climaticas:
                name = str(j) + '_lag_' + str(i)
                df[name] = df[j].shift(i)


    lag_number = 24*2
    portugal = ['TOTAL_DEMANDA_POR', 'TOTAL_PRODUCCION_POR'
                  ]
    for i in range(1, lag_number, 1):
            for j in portugal:
                name = str(j) + '_lag_' + str(i)
                df[name] = df[j].shift(i)

    df = df[df['PESPANIA'] > 0]
    df = df.dropna(how='any', axis=0)


    normal = df[df['DUMMY_5_DAY'] == 0]
    anormal = df[df['DUMMY_5_DAY'] == 1]

    del normal['DUMMY_5_DAY']
    del anormal['DUMMY_5_DAY']

    # NORMALIZE
    column_names = normal.columns.values.tolist()
    normal = preprocessing.scale(normal)
    normal = pd.DataFrame(normal, columns=[column_names])

    column_names = anormal.columns.values.tolist()
    anormal = preprocessing.scale(anormal)
    anormal = pd.DataFrame(anormal, columns=[column_names])

    total_values = len(df.index)
    print('total rows ', total_values)
    anormal_values = len(anormal.index)
    print('anormal rows ', anormal_values)

    proportion = anormal_values / total_values
    print('proportion of anormal ', proportion)

    normalY = normal[['PESPANIA']]
    normalX = normal
    del normalX['PESPANIA']

    anormalY = anormal[['PESPANIA']]
    anormalX = anormal
    del anormalX['PESPANIA']

    names = normalX.columns.values
    fileNames = np.array(names)

    # Solo tomamos test y train del normal con el mismo tamaño del test que la muestra de anormales
    X_train, X_test, y_train, y_test = train_test_split(normalX, normalY, test_size=proportion, random_state=42)

    nTreeList = range(2000, 2001, 1)
    for iTrees in nTreeList:
        tresholds = np.linspace(0.1, 1.0, 200)

        min_samples_leaf = round(len(X_train.index) * 0.01)
        print('min_samples_leaf ', min_samples_leaf)
        min_samples_split = min_samples_leaf*10
        print(iTrees)
        depth = 50
        maxFeat = (round((len(df.columns)/3)))
        print('Feature Set ', maxFeat)
        fileModel = ensemble.ExtraTreesRegressor(criterion='mse', bootstrap=False,
                                                 min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                 n_estimators=iTrees,
                                                 max_depth=depth, max_features=maxFeat, oob_score=False,
                                                 random_state=531, verbose=1)

        fileModel.fit(X_train.values, y_train.values)
        prediction_normal = fileModel.predict(X_test)

        print('MSE NORMAL ', mean_squared_error(y_test, prediction_normal))
        print('R2 NORMAL ', r2_score(y_test, prediction_normal))

        prediction_normal = pd.DataFrame(prediction_normal, index=y_test.index)
        prediction_normal = pd.concat([y_test, prediction_normal], axis=1)
        prediction_normal.columns = ['PESPANIA_REAL_NO_COLUSION', 'PESPANIA_PRED_NO_COLUSION']
        prediction_normal['DIF_PORC'] = (prediction_normal['PESPANIA_REAL_NO_COLUSION'] - prediction_normal[
            'PESPANIA_PRED_NO_COLUSION']) / prediction_normal['PESPANIA_PRED_NO_COLUSION']
        print('PRECIO PROMEDIO PREDICHO - NO COLUSION ', prediction_normal['PESPANIA_PRED_NO_COLUSION'].mean())
        print('PRECIO PROMEDIO REAL - NO COLUSION ', prediction_normal['PESPANIA_REAL_NO_COLUSION'].mean())
        print('DIFERENCIA PROMEDIO PORCENTUAL (REAL/PRED -1) ', prediction_normal['DIF_PORC'].mean() * 100, '%')
        prediction_normal.to_csv('prediction_normal.csv', sep=';', index=False)

        prediction_anormal = fileModel.predict(anormalX)

        print('MSE ANORMAL ', mean_squared_error(anormalY, prediction_anormal))
        print('R2 ANORMAL ', r2_score(anormalY, prediction_anormal))

        prediction_anormal = pd.DataFrame(prediction_anormal, index=anormalY.index)
        prediction_anormal = pd.concat([anormalY, prediction_anormal], axis=1)
        prediction_anormal.columns = ['PESPANIA_REAL_COLUSION', 'PESPANIA_PRED_COLUSION']
        prediction_anormal['DIF_PORC'] = (prediction_anormal['PESPANIA_REAL_COLUSION'] - prediction_anormal[
            'PESPANIA_PRED_COLUSION']) / prediction_anormal['PESPANIA_PRED_COLUSION']
        print('PRECIO PROMEDIO PREDICHO - COLUSION ', prediction_anormal['PESPANIA_PRED_COLUSION'].mean())
        print('PRECIO PROMEDIO REAL - COLUSION ', prediction_anormal['PESPANIA_REAL_COLUSION'].mean())
        print('DIFERENCIA PROMEDIO PORCENTUAL (REAL/PRED -1)', prediction_anormal['DIF_PORC'].mean() * 100, '%')
        prediction_anormal.to_csv('prediction_anormal.csv', sep=';', index=False)

        fig, ax = plot.subplots()
        sns.regplot(y='PESPANIA_PRED_COLUSION', x='PESPANIA_REAL_COLUSION', data=prediction_anormal, ax=ax,
                    label='COLUSION')
        sns.regplot(y='PESPANIA_PRED_NO_COLUSION', x='PESPANIA_REAL_NO_COLUSION', data=prediction_normal, ax=ax,
                    label='NON-COLUSION')
        diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label='perfect prediction')
        plot.legend(loc='best')
        plot.title('Differences between Prices using ERF')
        plot.show()

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()
        sorted_idx = np.argsort(featureImportance)
        barPos = np.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align='center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()



def ARIMAX_byday(df):

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

    df['ANIO'] = df['ANIO'].map(int)
    df['MES'] = df['MES'].map(int)
    df['DIA'] = df['DIA'].map(int)

    df = df.groupby(['FECHA']).mean().reset_index()
    # Log de Precios
    # df['PESPANIA'] = np.log(df['PESPANIA'])
    # df['PPORTUGAL'] = np.log(df['PPORTUGAL'])
    del df['PPORTUGAL']




    df.index.name = None
    df.reset_index(inplace=True)

    df['index'] = df['FECHA']
    df.set_index(['index'], inplace=True)
    df.index.name = None



    df['WEEKDAY'] = df['FECHA'].dt.dayofweek

    df['DUMMY_2010_REGIMEN'] = pd.Series(0, index=df.index)
    df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1

    del df['HORA']
    del df['FECHA']
    del df['FECHA_HORA']


    # TARGET VARIABLE
    dummy_important = ['DUMMY', 'DUMMY_5_DAY', 'DUMMY_10_DAY', 'DUMMY_15_DAY', 'DUMMY_20_DAY']
    for i in dummy_important:
        del df[i]

    # statistics_basic()

    # DUMMIES
    dummy_var = []
    for i in dummy_var:
        name = str(i)
        dummy = pd.get_dummies(df[i], prefix=name)
        df = pd.concat([df, dummy], axis=1)
        del dummy
        del df[i]

    # LAGS
    lag_number = 4
    lag_variables = ['TOTAL_DDA_ES']

    for i in range(1, lag_number, 1):
        for j in lag_variables:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 2
    climaticas = [
                  ]
    for i in range(1, lag_number, 1):
        for j in climaticas:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 3
    portugal = ['TOTAL_DEMANDA_POR'
                ]
    for i in range(1, lag_number, 1):
        for j in portugal:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    df = df[df['PESPANIA'] > 0]
    df = df.dropna(how='any', axis=0)


    # REDUCTION VAR

    df['VERANO-INVIERNO'] = pd.Series(0, index=df.index)
    df.loc[df['MES'].isin([1, 12]), 'VERANO-INVIERNO'] = 1

    df['WORKDAY'] = pd.Series(0, index=df.index)
    df.loc[df['WEEKDAY'].isin([0, 1, 2, 3, 4]), 'WORKDAY'] = 1

    reduction_variables = ['DIA', 'TOTAL_IMPORTACION_ES', 'TOTAL_EXPORTACIONES_ES', 'TOTAL_PRODUCCION_POR',
                           'TOTAL_PRODUCCION_ES', 'ANIO', 'MES', 'WEEKDAY', 'TOTAL_DEMANDA_NAC_ES', 'TME_BCN',
                           'TMAX_BCN', 'TMIN_BCN', 'PP_BCN', 'VERANO-INVIERNO', 'TOTAL_DEMANDA_POR_lag_2',
                           'TOTAL_DDA_ES_lag_3']

    for i in reduction_variables:
        del df[i]

    # Log Variables
    log_var = ['PESPANIA', 'TOTAL_DDA_ES',
               'TOTAL_POT_IND_ES', 'TOTAL_DEMANDA_POR', 'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO',
               'NUCLEAR', 'CARBON NACIONAL', 'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA',
               'REG_ESPECIAL', 'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM',
               'GDP', '%EOLICA', 'TOTAL_DDA_ES_lag_1', 'TOTAL_DDA_ES_lag_2',
               'TOTAL_DEMANDA_POR_lag_1']

    for i in log_var:
        df[df == 0] = 0.000001
        df[i] = np.log(df[i])

    print(len(df.columns))
    print(df.columns.values)
    model = sm.tsa.statespace.SARIMAX(df['PESPANIA'], exog=df.drop('PESPANIA', axis=1),
                                          order=(2, 1, 3), trend=None, enforce_invertibility=False,
                                          enforce_stationarity=False)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plot.show()
    residuals.plot(kind='kde')
    plot.show()
    print(residuals.describe())

def statistics_basic():

    df.index.name = None
    df.reset_index(inplace=True)
    df['FECHA_HORA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str) + ' '\
                       + df['HORA'].map(str) + ':00:00'

    df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA'], format='%Y-%m-%d')
    df['index'] = df['FECHA_HORA']
    df.set_index(['index'], inplace=True)
    df.index.name = None

    # PLOT PRICE SERIES--------------------------------------------------------------

    df[['FECHA_HORA', 'PESPANIA']].set_index('FECHA_HORA').plot()
    plot.savefig('precio_espania.png')
    plot.close()

    '''
    df[['FECHA_HORA', 'PPORTUGAL']].set_index('FECHA_HORA').plot()
    plot.savefig('log_precio_portugal.png')
    '''

    decomposition = seasonal_decompose(df['PESPANIA'], freq=12)
    fig = plot.figure()
    fig = decomposition.plot()
    fig.set_size_inches(15, 8)
    plot.savefig('precio_espania_decompose.png')
    plot.close()

    # EXISTENCIA DE CORRELACION SERIAL-------------------------------------------------

    # AUTOCORRELATION PLOT
    serie = df[['PESPANIA']]
    autocorrelation_plot(serie)
    plot.savefig('autocorr_pespania.png')
    plot.close()

    # ESTACIONARIEDAD-----------------------------------------------------------------




    def test_stationarity(timeseries):
        # Determing rolling statistics
        rolmean = pd.rolling_mean(timeseries, window=12)
        rolstd = pd.rolling_std(timeseries, window=12)

        # Plot rolling statistics:
        fig = plot.figure(figsize=(12, 8))
        orig = plot.plot(timeseries, color='blue', label='Original')
        mean = plot.plot(rolmean, color='red', label='Rolling Mean')
        std = plot.plot(rolstd, color='black', label='Rolling Std')
        plot.legend(loc='best')
        plot.title('Rolling Mean & Standard Deviation')
        plot.savefig('stationarity_pespania.png')
        plot.close()

        # Perform Dickey-Fuller test:
        '''
        if the ‘Test Statistic’ is greater than the ‘Critical Value’ than the time series is stationary. 
        '''
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    test_stationarity(df['PESPANIA'])

    '''
    En este caso la serie es estacionaria por lo que no necesitamos usar esto de abajo ni diferenciar I = 0
    Si necesita diferenciarse:
    df['D_PESPANIA'] = df['PESPANIA'] - df['PESPANIA'].shift(1)
    test_stationarity(df['D_PESPANIA'])
    
    Para remover la seasonality of the data
    anio = 365*24
    df['SEASONAL_DIF'] = df['PESPANIA'] - df['PESPANIA'].shift(anio)
    test_stationarity(df['SEASONAL_DIF'])
    '''


    # ELEGIR EL NUMERO OPTIMO DE AR y MA

    # ACF AND PACF
    fig = plot.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df['PESPANIA'], lags=10000, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df['PESPANIA'], lags=65, ax=ax2)
    plot.savefig('pac_ac_PESPANIA.png')
    plot.close()

    # When Autocorrelation MA
    # Partial for AR



# extreme_random_byday(df)
# ARIMAX_byday(df)

# extreme_random_byhour(df)

