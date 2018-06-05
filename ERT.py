import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plot
import seaborn as sns
import STRING
import datetime

def extreme_random_byday(df, evaluate_var='DUMMY_30_DAY'):
    # Log de Precios
    # df['PESPANIA'] = np.log(df['PESPANIA'])
    # df['PPORTUGAL'] = np.log(df['PPORTUGAL'])
    del df['PPORTUGAL']

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

    df['WEEKDAY'] = df['FECHA'].dt.dayofweek
    

    # df['DUMMY_2010_REGIMEN'] = pd.Series(0, index=df.index)
    # df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1
    # df = df[df['FECHA'] >= '2010-01-01']
    
    df['ANIO'] = df['ANIO'].map(int)
    df['MES'] = df['MES'].map(int)
    df['DIA'] = df['DIA'].map(int)

    df = df.groupby(['FECHA']).mean().reset_index()

    del df['HORA']
    del df['FECHA']
    del df['FECHA_HORA']
    del df['DIA']

    # TARGET VARIABLE

    dummy_important = ['DUMMY', 'DUMMY_5_DAY', 'DUMMY_10_DAY', 'DUMMY_15_DAY', 'DUMMY_20_DAY', 'DUMMY_30_DAY']
    dummy_important.remove(evaluate_var)
    for i in dummy_important:
        del df[i]

    # DIFFERENCIATE

    # DIFERENCIA PESPANIA
    '''
    df['PESPANIA'] = df['PESPANIA'] - df['PESPANIA'].shift(1)
    df = df.dropna(axis=0)

    # DIFERENCIA RESTO
    
    need_differenciation = ['TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR', 'CICLO_COMBINADO', 'FUEL_PRIMA',
                            'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME_MADRID', 'TMAX_MADRID', 'TME_BCN',
                            'TMAX_BCN', 'TMIN_BCN', 'GDP']


    for i in need_differenciation:
        name = 'D_' + str(i)
        df[name] = df[i] - df[i].shift(1)
        del df[i]

    df = df.dropna()
    '''

    # DUMMIES
    dummy_var = ['ANIO', 'MES', 'WEEKDAY']
    for i in dummy_var:
        name = str(i)
        dummy = pd.get_dummies(df[i], prefix=name)
        df = pd.concat([df, dummy], axis=1)
        del dummy
        del df[i]

    # LAGS
    lag_AR = 28
    for i in range(1, lag_AR+1, 1):
        name = 'PESPANIA_lag_' + str(i)
        df[name] = df['PESPANIA'].shift(i)

    lag_number = 7
    lag_variables = ['TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES',
                     'TOTAL_EXPORTACIONES_ES', 'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES',
                     'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                     'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL', 
                     'PRICE_OIL',
                     'PRICE_GAS', 'RISK_PREMIUM']

    for i in range(1, lag_number, 1):
        for j in lag_variables:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 1
    climaticas = ['TME_MADRID', 'TMAX_MADRID', 'TMIN_MADRID', 'PP_MADRID', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN', 
                  'PP_BCN'
                  ]
    for i in range(1, lag_number+1, 1):
        for j in climaticas:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 2
    portugal = ['TOTAL_DEMANDA_POR', 'TOTAL_PRODUCCION_POR'
                ]
    for i in range(1, lag_number+1, 1):
        for j in portugal:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)


    df = df.dropna(how='any', axis=0)

    normal = df[df[evaluate_var] == 0]
    anormal = df[df[evaluate_var] == 1]

    del normal[evaluate_var]
    del anormal[evaluate_var]

    # NORMALIZE
    column_names = normal.columns.values.tolist()
    normal = preprocessing.robust_scale(normal)
    normal = pd.DataFrame(normal, columns=[column_names])

    column_names = anormal.columns.values.tolist()
    anormal = preprocessing.robust_scale(anormal)
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

        min_samples_leaf = round(len(X_train.index) * 0.005)
        print('min_samples_leaf ', min_samples_leaf)
        min_samples_split = min_samples_leaf * 10
        print('min_samples_split ', min_samples_split)
        print('iTrees ', iTrees)
        depth = 50
        maxFeat = (round((len(df.columns) / 3)))
        print('Feature Set ', maxFeat)

        fileModel = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       max_depth=depth, verbose=1)

        fileModel1 = ensemble.ExtraTreesRegressor(criterion='mse', bootstrap=False,
                                                  min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                  n_estimators=iTrees,
                                                  max_depth=depth, max_features=maxFeat, oob_score=False,
                                                  random_state=531, verbose=1)

        fileModel2 = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf, verbose=1, max_features=maxFeat)

        fileModel.fit(X_train.values, y_train.values)
        prediction_normal = fileModel.predict(X_test)

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
        prediction_normal.to_csv('prediction_normal_dia.csv', sep=';', index=False)

        prediction_anormal = fileModel.predict(anormalX)

        print('MSE ANORMAL ', mean_squared_error(anormalY, prediction_anormal))
        print('R2 ANORMAL ', r2_score(anormalY, prediction_anormal))

        prediction_anormal = pd.DataFrame(prediction_anormal, index=anormalY.index)
        prediction_anormal = pd.concat([anormalY, prediction_anormal], axis=1)
        prediction_anormal.columns = ['PESPANIA_REAL_COLUSION', 'PESPANIA_PRED_COLUSION']
        prediction_anormal['DIF_PORC'] = (prediction_anormal['PESPANIA_REAL_COLUSION'] - prediction_anormal[
            'PESPANIA_PRED_COLUSION']) / prediction_anormal['PESPANIA_PRED_COLUSION']
        print('PRECIO PROMEDIO PREDICHO - COLUSION %.5f' % prediction_anormal['PESPANIA_PRED_COLUSION'].mean())
        print('PRECIO PROMEDIO REAL - COLUSION %.5f' % prediction_anormal['PESPANIA_REAL_COLUSION'].mean())
        print('DIFERENCIA PROMEDIO PORCENTUAL (REAL/PRED -1)', prediction_anormal['DIF_PORC'].mean() * 100, '%')
        prediction_anormal.to_csv('prediction_anormal_dia.csv', sep=';', index=False)

        fig, ax = plot.subplots()
        sns.regplot(y='PESPANIA_PRED_COLUSION', x='PESPANIA_REAL_COLUSION', data=prediction_anormal, ax=ax,
                    label='COLUSION')
        sns.regplot(y='PESPANIA_PRED_NO_COLUSION', x='PESPANIA_REAL_NO_COLUSION', data=prediction_normal, ax=ax,
                    label='NON-COLUSION')
        diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label='perfect prediction')
        plot.legend(loc='best')
        plot.title('Differences between Prices using ERF')
        plot.show()

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

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()
        sorted_idx = np.argsort(featureImportance)
        fi = featureImportance[sorted_idx]
        fi = fi[-10:]
        barPos = np.arange(sorted_idx.shape[0]) + 0.5
        barPos = barPos[-10:]
        plot.barh(barPos, fi, align='center')
        fileNames = fileNames[sorted_idx]
        fileNames = fileNames[-10:]
        plot.yticks(barPos, fileNames)
        plot.xlabel('Variable Importance')
        plot.show()


def extreme_random_byhour(df, evaluate_var='DUMMY_30_DAY'):
    # Log de Precios
    # df['PESPANIA'] = np.log(df['PESPANIA'])
    # df['PPORTUGAL'] = np.log(df['PPORTUGAL'])
    del df['PPORTUGAL']

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

    df['WEEKDAY'] = df['FECHA'].dt.dayofweek

    # df['DUMMY_2010_REGIMEN'] = pd.Series(0, index=df.index)
    # df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1
    # df = df[df['FECHA'] >= '2010-01-01']

    df['ANIO'] = df['ANIO'].map(int)
    df['MES'] = df['MES'].map(int)
    df['DIA'] = df['DIA'].map(int)

    # df = df.groupby(['FECHA']).mean().reset_index()

    del df['HORA']
    del df['FECHA']
    del df['FECHA_HORA']
    del df['DIA']

    # TARGET VARIABLE

    dummy_important = ['DUMMY', 'DUMMY_5_DAY', 'DUMMY_10_DAY', 'DUMMY_15_DAY', 'DUMMY_20_DAY', 'DUMMY_30_DAY']
    dummy_important.remove(evaluate_var)
    for i in dummy_important:
        del df[i]

    # DIFFERENCIATE

    # DIFERENCIA PESPANIA
    '''
    df['PESPANIA'] = df['PESPANIA'] - df['PESPANIA'].shift(1)
    df = df.dropna(axis=0)

    # DIFERENCIA RESTO

    need_differenciation = ['TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR', 'CICLO_COMBINADO', 'FUEL_PRIMA',
                            'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME_MADRID', 'TMAX_MADRID', 'TME_BCN',
                            'TMAX_BCN', 'TMIN_BCN', 'GDP']


    for i in need_differenciation:
        name = 'D_' + str(i)
        df[name] = df[i] - df[i].shift(1)
        del df[i]

    df = df.dropna()
    '''

    # DUMMIES
    dummy_var = ['ANIO', 'MES', 'WEEKDAY']
    for i in dummy_var:
        name = str(i)
        dummy = pd.get_dummies(df[i], prefix=name)
        df = pd.concat([df, dummy], axis=1)
        del dummy
        del df[i]

    # LAGS
    lag_AR = 28
    for i in range(1, lag_AR + 1, 1):
        name = 'PESPANIA_lag_' + str(i)
        df[name] = df['PESPANIA'].shift(i)

    lag_number = 24
    lag_variables = ['TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES',
                     'TOTAL_EXPORTACIONES_ES', 'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES',
                     'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                     'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL',
                     'PRICE_OIL',
                     'PRICE_GAS', 'RISK_PREMIUM']

    for i in range(1, lag_number, 1):
        for j in lag_variables:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 24
    climaticas = ['TME_MADRID', 'TMAX_MADRID', 'TMIN_MADRID', 'PP_MADRID', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN',
                  'PP_BCN'
                  ]
    for i in range(1, lag_number + 1, 1):
        for j in climaticas:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 24
    portugal = ['TOTAL_DEMANDA_POR', 'TOTAL_PRODUCCION_POR'
                ]
    for i in range(1, lag_number + 1, 1):
        for j in portugal:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    df = df.dropna(how='any', axis=0)

    normal = df[df[evaluate_var] == 0]
    anormal = df[df[evaluate_var] == 1]

    del normal[evaluate_var]
    del anormal[evaluate_var]

    # NORMALIZE
    column_names = normal.columns.values.tolist()
    normal = preprocessing.robust_scale(normal)
    normal = pd.DataFrame(normal, columns=[column_names])

    column_names = anormal.columns.values.tolist()
    anormal = preprocessing.robust_scale(anormal)
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

        min_samples_leaf = round(len(X_train.index) * 0.005)
        print('min_samples_leaf ', min_samples_leaf)
        min_samples_split = min_samples_leaf * 10
        print('min_samples_split ', min_samples_split)
        print('iTrees ', iTrees)
        depth = 50
        maxFeat = (round((len(df.columns) / 3)))
        print('Feature Set ', maxFeat)

        fileModel = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       max_depth=depth, verbose=1)

        fileModel1 = ensemble.ExtraTreesRegressor(criterion='mse', bootstrap=False,
                                                  min_samples_leaf=min_samples_leaf,
                                                  min_samples_split=min_samples_split,
                                                  n_estimators=iTrees,
                                                  max_depth=depth, max_features=maxFeat, oob_score=False,
                                                  random_state=531, verbose=1)

        fileModel2 = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf, verbose=1, max_features=maxFeat)

        fileModel.fit(X_train.values, y_train.values)
        prediction_normal = fileModel.predict(X_test)

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
        prediction_normal.to_csv('prediction_normal_dia.csv', sep=';', index=False)

        prediction_anormal = fileModel.predict(anormalX)

        print('MSE ANORMAL ', mean_squared_error(anormalY, prediction_anormal))
        print('R2 ANORMAL ', r2_score(anormalY, prediction_anormal))

        prediction_anormal = pd.DataFrame(prediction_anormal, index=anormalY.index)
        prediction_anormal = pd.concat([anormalY, prediction_anormal], axis=1)
        prediction_anormal.columns = ['PESPANIA_REAL_COLUSION', 'PESPANIA_PRED_COLUSION']
        prediction_anormal['DIF_PORC'] = (prediction_anormal['PESPANIA_REAL_COLUSION'] - prediction_anormal[
            'PESPANIA_PRED_COLUSION']) / prediction_anormal['PESPANIA_PRED_COLUSION']
        print('PRECIO PROMEDIO PREDICHO - COLUSION %.5f' % prediction_anormal['PESPANIA_PRED_COLUSION'].mean())
        print('PRECIO PROMEDIO REAL - COLUSION %.5f' % prediction_anormal['PESPANIA_REAL_COLUSION'].mean())
        print('DIFERENCIA PROMEDIO PORCENTUAL (REAL/PRED -1)', prediction_anormal['DIF_PORC'].mean() * 100, '%')
        prediction_anormal.to_csv('prediction_anormal_dia.csv', sep=';', index=False)

        fig, ax = plot.subplots()
        sns.regplot(y='PESPANIA_PRED_COLUSION', x='PESPANIA_REAL_COLUSION', data=prediction_anormal, ax=ax,
                    label='COLUSION')
        sns.regplot(y='PESPANIA_PRED_NO_COLUSION', x='PESPANIA_REAL_NO_COLUSION', data=prediction_normal, ax=ax,
                    label='NON-COLUSION')
        diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label='perfect prediction')
        plot.legend(loc='best')
        plot.title('Differences between Prices using ERF')
        plot.show()

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

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()
        sorted_idx = np.argsort(featureImportance)
        fi = featureImportance[sorted_idx]
        fi = fi[-10:]
        barPos = np.arange(sorted_idx.shape[0]) + 0.5
        barPos = barPos[-10:]
        plot.barh(barPos, fi, align='center')
        fileNames = fileNames[sorted_idx]
        fileNames = fileNames[-10:]
        plot.yticks(barPos, fileNames)
        plot.xlabel('Variable Importance')
        plot.show()


def ert_by_day_by_auction(df, evaluate_var):
    # Log de Precios
    # df['PESPANIA'] = np.log(df['PESPANIA'])
    # df['PPORTUGAL'] = np.log(df['PPORTUGAL'])
    del df['PPORTUGAL']

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

    df['WEEKDAY'] = df['FECHA'].dt.dayofweek

    # df['DUMMY_2010_REGIMEN'] = pd.Series(0, index=df.index)
    # df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1
    # df = df[df['FECHA'] >= '2010-01-01']

    df['ANIO'] = df['ANIO'].map(int)
    df['MES'] = df['MES'].map(int)
    df['DIA'] = df['DIA'].map(int)

    df = df.groupby(['FECHA']).mean().reset_index()

    del df['HORA']
    del df['FECHA_HORA']
    del df['DIA']

    # TARGET VARIABLE

    dummy_important = ['DUMMY_5_DAY', 'DUMMY_10_DAY', 'DUMMY_15_DAY', 'DUMMY_20_DAY', 'DUMMY_30_DAY']
    dummy_important.remove(evaluate_var)
    for i in dummy_important:
        del df[i]

    # DIFFERENCIATE

    # DIFERENCIA PESPANIA
    '''
    df['PESPANIA'] = df['PESPANIA'] - df['PESPANIA'].shift(1)
    df = df.dropna(axis=0)

    # DIFERENCIA RESTO

    need_differenciation = ['TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR', 'CICLO_COMBINADO', 'FUEL_PRIMA',
                            'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME_MADRID', 'TMAX_MADRID', 'TME_BCN',
                            'TMAX_BCN', 'TMIN_BCN', 'GDP']


    for i in need_differenciation:
        name = 'D_' + str(i)
        df[name] = df[i] - df[i].shift(1)
        del df[i]

    df = df.dropna()
    '''

    # DUMMIES
    dummy_var = ['ANIO', 'MES', 'WEEKDAY']
    for i in dummy_var:
        name = str(i)
        dummy = pd.get_dummies(df[i], prefix=name)
        df = pd.concat([df, dummy], axis=1)
        del dummy
        del df[i]

    # LAGS
    lag_AR = 28
    for i in range(1, lag_AR + 1, 1):
        name = 'PESPANIA_lag_' + str(i)
        df[name] = df['PESPANIA'].shift(i)

    lag_number = 7
    lag_variables = ['TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES',
                     'TOTAL_EXPORTACIONES_ES', 'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES',
                     'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                     'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL',
                     'PRICE_OIL',
                     'PRICE_GAS', 'RISK_PREMIUM']

    for i in range(1, lag_number, 1):
        for j in lag_variables:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 1
    climaticas = ['TME_MADRID', 'TMAX_MADRID', 'TMIN_MADRID', 'PP_MADRID', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN',
                  'PP_BCN'
                  ]
    for i in range(1, lag_number + 1, 1):
        for j in climaticas:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 2
    portugal = ['TOTAL_DEMANDA_POR', 'TOTAL_PRODUCCION_POR'
                ]
    for i in range(1, lag_number + 1, 1):
        for j in portugal:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    df = df.dropna(how='any', axis=0)

    normal = df[df[evaluate_var] == 0]
    anormal = df[df[evaluate_var] == 1]

    del normal[evaluate_var]
    del anormal[evaluate_var]

    # NORMALIZE
    column_names = normal.columns.values.tolist()
    normal_date = normal[['FECHA']]
    normal = preprocessing.robust_scale(normal.drop('FECHA', axis=1).values)
    normal = pd.DataFrame(normal)
    normal = pd.concat([normal, normal_date], axis=1)
    normal = pd.DataFrame(normal, columns=[column_names])

    column_names = anormal.columns.values.tolist()
    anormal_date = anormal[['FECHA']]
    anormal = preprocessing.robust_scale(anormal.drop('FECHA', axis=1).values)
    anormal = pd.DataFrame(anormal)
    anormal = pd.concat([anormal, anormal_date], axis=1)
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

    auction_date = df.FECHA[df['DUMMY'] == 1].tolist()

    del df['DUMMY']

    days_before = 30

    for auction_date in auction_date:

        anormalY = df[df['FECHA'] >= auction_date - datetime.timedelta(days=days_before)]
        anormalY = anormalY[anormalY['FECHA'] <= auction_date]
        anormalY = anormalY[['PESPANIA']]

        df_before_auction = df[df['FECHA'] < auction_date - datetime.timedelta(days=days_before)]
        df_before_auction_Y = df_before_auction[['PESPANIA']]
        df_before_auction_X = df_before_auction.drop(['FECHA', 'PESPANIA'], axis=1)

        normalY = df[df['FECHA'] >= auction_date]
        normalY = normalY[normalY['FECHA'] <= auction_date + datetime.timedelta(days=days_before)]
        future_dates = normalY.drop(['FECHA', 'PESPANIA'], axis=1)
        normalY = normalY[['PESPANIA']]

        df_before_auction1 = df[df['FECHA'] <= auction_date]
        df_before_auction_Y_1 = df_before_auction1[['PESPANIA']]
        df_before_auction_X_1 = df_before_auction1.drop(['FECHA', 'PESPANIA'], axis=1)


        min_samples_leaf = round(len(df_before_auction.index) * 0.005)
        print('min_samples_leaf ', min_samples_leaf)
        min_samples_split = min_samples_leaf * 10
        print('min_samples_split ', min_samples_split)
        iTrees = 100
        print('iTrees ', iTrees)
        depth = 50
        maxFeat = (round((len(df.columns) / 3)))
        print('Feature Set ', maxFeat)

        fileModel = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=iTrees,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       max_depth=depth, verbose=1)

        fileModel.fit(df_before_auction_X, df_before_auction_Y)

        fileModel2 = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=iTrees,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       max_depth=depth, verbose=1)
        fileModel2.fit(df_before_auction_X_1, df_before_auction_Y_1)

        # Predecimos los siguientes días para el anormal
        y_hat = fileModel.predict(future_dates)
        y_hat = pd.DataFrame(y_hat, columns=['yhat'])
        prediction_anormal = y_hat
        anormalY = anormalY.reset_index(drop=True)

        # Predecimos los siguientes días para el normal
        y_hat = fileModel2.predict(future_dates)
        y_hat = pd.DataFrame(y_hat, columns=['yhat'])
        prediction_normal = y_hat
        y_test = normalY.reset_index(drop=True)

        print('MSE ANORMAL ', mean_squared_error(anormalY, prediction_anormal))
        print('R2 ANORMAL ', r2_score(anormalY, prediction_anormal))

        # Comparamos con el verdadero valor

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



def ert_by_hour_by_auction(df, evaluate_var):
    # Log de Precios
    # df['PESPANIA'] = np.log(df['PESPANIA'])
    # df['PPORTUGAL'] = np.log(df['PPORTUGAL'])
    del df['PPORTUGAL']

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

    df['WEEKDAY'] = df['FECHA'].dt.dayofweek

    # df['DUMMY_2010_REGIMEN'] = pd.Series(0, index=df.index)
    # df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1
    # df = df[df['FECHA'] >= '2010-01-01']

    df['ANIO'] = df['ANIO'].map(int)
    df['MES'] = df['MES'].map(int)
    df['DIA'] = df['DIA'].map(int)

    # df = df.groupby(['FECHA']).mean().reset_index()

    del df['HORA']
    del df['FECHA_HORA']
    del df['DIA']

    # TARGET VARIABLE

    dummy_important = ['DUMMY_5_DAY', 'DUMMY_10_DAY', 'DUMMY_15_DAY', 'DUMMY_20_DAY', 'DUMMY_30_DAY']
    dummy_important.remove(evaluate_var)
    for i in dummy_important:
        del df[i]

    # DIFFERENCIATE

    # DIFERENCIA PESPANIA
    '''
    df['PESPANIA'] = df['PESPANIA'] - df['PESPANIA'].shift(1)
    df = df.dropna(axis=0)

    # DIFERENCIA RESTO

    need_differenciation = ['TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR', 'CICLO_COMBINADO', 'FUEL_PRIMA',
                            'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME_MADRID', 'TMAX_MADRID', 'TME_BCN',
                            'TMAX_BCN', 'TMIN_BCN', 'GDP']


    for i in need_differenciation:
        name = 'D_' + str(i)
        df[name] = df[i] - df[i].shift(1)
        del df[i]

    df = df.dropna()
    '''

    # DUMMIES
    dummy_var = ['ANIO', 'MES', 'WEEKDAY']
    for i in dummy_var:
        name = str(i)
        dummy = pd.get_dummies(df[i], prefix=name)
        df = pd.concat([df, dummy], axis=1)
        del dummy
        del df[i]

    # LAGS
    lag_AR = 24
    for i in range(1, lag_AR + 1, 1):
        name = 'PESPANIA_lag_' + str(i)
        df[name] = df['PESPANIA'].shift(i)

    lag_number = 24
    lag_variables = ['TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES',
                     'TOTAL_EXPORTACIONES_ES', 'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES',
                     'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                     'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL',
                     'PRICE_OIL',
                     'PRICE_GAS', 'RISK_PREMIUM']

    for i in range(1, lag_number, 1):
        for j in lag_variables:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 24
    climaticas = ['TME_MADRID', 'TMAX_MADRID', 'TMIN_MADRID', 'PP_MADRID', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN',
                  'PP_BCN'
                  ]
    for i in range(1, lag_number + 1, 1):
        for j in climaticas:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    lag_number = 24
    portugal = ['TOTAL_DEMANDA_POR', 'TOTAL_PRODUCCION_POR'
                ]
    for i in range(1, lag_number + 1, 1):
        for j in portugal:
            name = str(j) + '_lag_' + str(i)
            df[name] = df[j].shift(i)

    df = df.dropna(how='any', axis=0)

    normal = df[df[evaluate_var] == 0]
    anormal = df[df[evaluate_var] == 1]

    del normal[evaluate_var]
    del anormal[evaluate_var]

    # NORMALIZE
    column_names = normal.columns.values.tolist()
    normal_date = normal[['FECHA']]
    normal = preprocessing.robust_scale(normal.drop('FECHA', axis=1).values)
    normal = pd.DataFrame(normal)
    normal = pd.concat([normal, normal_date], axis=1)
    normal = pd.DataFrame(normal, columns=[column_names])

    column_names = anormal.columns.values.tolist()
    anormal_date = anormal[['FECHA']]
    anormal = preprocessing.robust_scale(anormal.drop('FECHA', axis=1).values)
    anormal = pd.DataFrame(anormal)
    anormal = pd.concat([anormal, anormal_date], axis=1)
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

    auction_date = df.FECHA[df['DUMMY'] == 1].tolist()

    del df['DUMMY']

    days_before = 30

    for auction_date in auction_date:

        anormalY = df[df['FECHA'] >= auction_date - datetime.timedelta(days=days_before)]
        anormalY = anormalY[anormalY['FECHA'] <= auction_date]
        anormalY = anormalY[['PESPANIA']]

        df_before_auction = df[df['FECHA'] < auction_date - datetime.timedelta(days=days_before)]
        df_before_auction_Y = df_before_auction[['PESPANIA']]
        df_before_auction_X = df_before_auction.drop(['FECHA', 'PESPANIA'], axis=1)

        normalY = df[df['FECHA'] >= auction_date]
        normalY = normalY[normalY['FECHA'] <= auction_date + datetime.timedelta(days=days_before)]
        future_dates = normalY.drop(['FECHA', 'PESPANIA'], axis=1)
        normalY = normalY[['PESPANIA']]

        df_before_auction1 = df[df['FECHA'] <= auction_date]
        df_before_auction_Y_1 = df_before_auction1[['PESPANIA']]
        df_before_auction_X_1 = df_before_auction1.drop(['FECHA', 'PESPANIA'], axis=1)


        min_samples_leaf = round(len(df_before_auction.index) * 0.005)
        print('min_samples_leaf ', min_samples_leaf)
        min_samples_split = min_samples_leaf * 10
        print('min_samples_split ', min_samples_split)
        iTrees = 100
        print('iTrees ', iTrees)
        depth = 50
        maxFeat = (round((len(df.columns) / 3)))
        print('Feature Set ', maxFeat)

        fileModel = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=iTrees,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       max_depth=depth, verbose=1)

        fileModel.fit(df_before_auction_X, df_before_auction_Y)

        fileModel2 = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=iTrees,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       max_depth=depth, verbose=1)
        fileModel2.fit(df_before_auction_X_1, df_before_auction_Y_1)

        # Predecimos los siguientes días para el anormal
        y_hat = fileModel.predict(future_dates)
        y_hat = pd.DataFrame(y_hat, columns=['yhat'])
        prediction_anormal = y_hat
        anormalY = anormalY.reset_index(drop=True)

        # Predecimos los siguientes días para el normal
        y_hat = fileModel2.predict(future_dates)
        y_hat = pd.DataFrame(y_hat, columns=['yhat'])
        prediction_normal = y_hat
        y_test = normalY.reset_index(drop=True)

        print('MSE ANORMAL ', mean_squared_error(anormalY, prediction_anormal))
        print('R2 ANORMAL ', r2_score(anormalY, prediction_anormal))

        # Comparamos con el verdadero valor

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

if __name__ == '__main__':
    sns.set(color_codes=True)
    df = pd.read_csv(STRING.final_file, sep=';', dtype={'FECHA': str, 'ANIO': str, 'MES': str, 'DIA': str})
    # extreme_random_byhour(df, evaluate_var='DUMMY_30_DAY')
    ert_by_hour_by_auction(df, 'DUMMY_30_DAY')
