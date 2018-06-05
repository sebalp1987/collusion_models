import seaborn as sns
import pandas as pd
import STRING
import numpy as np
import matplotlib.pyplot as plot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, pacf, acf
import stepwise_reg
import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

def process_df(df):

    del df['HORA']
    del df['FECHA_HORA']

    df.index.name = None
    df.reset_index(inplace=True)

    df['FECHA'] = df['ANIO'].map(str) + '-' + df['MES'].map(str) + '-' + df['DIA'].map(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')
    df['ANIO'] = df['ANIO'].map(int)
    df['MES'] = df['MES'].map(int)
    df['DIA'] = df['DIA'].map(int)

    df['DUMMY_2010_REGIMEN'] = pd.Series(0, index=df.index)
    df.loc[df['FECHA'] >= '2010-01-01', 'DUMMY_2010_REGIMEN'] = 1

    df = df.groupby(['FECHA']).mean().reset_index()


    df['index'] = df['FECHA']
    df.set_index(['index'], inplace=True)
    df.index.name = None

    df['WEEKDAY'] = df['FECHA'].dt.dayofweek

    df['VERANO-INVIERNO'] = pd.Series(0, index=df.index)
    df.loc[df['MES'].isin([1, 12]), 'VERANO-INVIERNO'] = 1

    df['WORKDAY'] = pd.Series(0, index=df.index)
    df.loc[df['WEEKDAY'].isin([0, 1, 2, 3, 4]), 'WORKDAY'] = 1
    del df['WEEKDAY']

    log_var = ['PESPANIA', 'TOTAL_DDA_ES',
               'TOTAL_POT_IND_ES', 'TOTAL_DEMANDA_POR', 'HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO',
               'NUCLEAR', 'CARBON NACIONAL', 'CARBON_IMPO', 'CICLO_COMBINADO', 'FUEL_SIN_PRIMA', 'FUEL_PRIMA',
               'REG_ESPECIAL', 'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM',
               'GDP', '%EOLICA', 'TOTAL_EXPORTACIONES_ES', 'TOTAL_PRODUCCION_POR',
                           'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES']



    return df

def plot_basics(df):

    # PLOT PRICE SERIES--------------------------------------------------------------

    df[['FECHA', 'PESPANIA']].set_index('FECHA').plot()
    plot.savefig('precio_espania_diario.png')
    plot.close()

    '''
    df[['FECHA_HORA', 'PPORTUGAL']].set_index('FECHA_HORA').plot()
    plot.savefig('log_precio_portugal.png')
    '''

    decomposition = seasonal_decompose(df['PESPANIA'], freq=12)
    fig = plot.figure()
    fig = decomposition.plot()
    fig.set_size_inches(15, 8)
    plot.savefig('precio_espania_decompose_dia.png')
    plot.close()

    # EXISTENCIA DE CORRELACION SERIAL-------------------------------------------------

    # AUTOCORRELATION PLOT
    serie = df[['PESPANIA']]
    autocorrelation_plot(serie)
    plot.savefig('autocorr_pespania_dia.png')
    plot.close()




    '''
    En este caso la serie es estacionaria por lo que no necesitamos usar esto de abajo ni diferenciar I = 0
    Si necesita diferenciarse:
    Para remover la seasonality of the data
    anio = 365*24
    df['SEASONAL_DIF'] = df['PESPANIA'] - df['PESPANIA'].shift(anio)
    test_stationarity(df['SEASONAL_DIF'])
    '''



def test_stationarity(timeseries, plot_show=False, plot_name='stationarity_pespania_dia.png'):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    if plot_show:
        fig = plot.figure(figsize=(12, 8))
        orig = plot.plot(timeseries, color='blue', label='Original')
        mean = plot.plot(rolmean, color='red', label='Rolling Mean')
        std = plot.plot(rolstd, color='black', label='Rolling Std')
        plot.legend(loc='best')
        plot.title('Rolling Mean & Standard Deviation')
        plot.savefig(plot_name)
        plot.close()

    # Perform Dickey-Fuller test:
    '''
    if the ‘Test Statistic’ is greater than the ‘Critical Value’ than the time series is stationary. 
    '''
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def granger_causality(df, evaluation_variable, dependant_variable='D_PESPANIA'):
    """
    The Null hypothesis for grangercausalitytests is that the time series in the second column, 
    x2, does NOT Granger cause the time series in the first column, x1. 
    Grange causality means that past values of x2 have a statistically significant 
    effect on the current value of x1, taking past values of x1 into account as regressors. 
    We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a 
    desired size of the test.
    
    :param df: 
    :param dependant_variable: 
    :param evaluation_variable: 
    :return: 
    """

    df1 = df[[evaluation_variable, dependant_variable]].values
    df2 = df[[dependant_variable, evaluation_variable]].values
    print('Y determina X')
    grangercausalitytests(df1, maxlag=5)
    print('X determina Y')
    grangercausalitytests(df2, maxlag=5)


def correlation_get_all(df: pd.DataFrame, get_all=False, get_specific='D_PESPANIA', output_file=False, show_plot=False):
    """
        We get correlations for a specific column or the whole dataframe.
        :param get_all: True if we want the whole dataframe correlation.
        :param get_specific: Column of the df we want to correlate.
        :param output_file: If we want a csv with the output.
        :param show_plot: If we want a Heat Map.
        :return: df correlations.
        """
    import seaborn as sns
    if get_all:
        corrmat = df.corr()
    else:
        index_i = df.columns.get_loc(get_specific)
        corrmat = df.corr().iloc[:, index_i]
    print(corrmat)

    if output_file:
        corrmat.to_csv('corrmat.csv', sep=';')

    if show_plot:
        f, ax = plot.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)
        networks = corrmat.columns.values.tolist()
        for i, network in enumerate(networks):
            if i and network != networks[i - 1]:
                ax.axhline(len(networks) - i, c="w")
                ax.axvline(i, c="w")
        f.tight_layout()
        plot.yticks(rotation=0)
        plot.xticks(rotation=90)
        plot.show()

def get_residuals(X, Y):
    error = []
    fileModel = linear_model.LinearRegression()
    fileModel.fit(X, Y)
    prediction = fileModel.predict(X)

    for i in range(len(Y)):
        error.append(Y[i] - prediction[i])

    error = pd.DataFrame(error, columns=['error'])
    return error


def serial_correlation(variable, plot_name='autocorr_error.png'):
    autocorrelation_plot(variable)
    plot.savefig(plot_name)
    plot.close()
    # https://robjhyndman.com/hyndsight/ljung-box-test/
    lags = min(10, round(len(variable)/5))
    print(acorr_ljungbox(variable, lags=lags))

def selecting_AR_MA(variable, lags=100, plot_name='PACF_ACF_D_PESPANIA.png'):
    pacf_result = pacf(variable, nlags=lags)
    acf_result = acf(variable, nlags=lags)

    print('PACF (AR) ', pacf_result)
    print('ACF (MA) ', acf_result)

    # ACF AND PACF
    fig = plot.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(variable, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(variable, lags=lags, ax=ax2)
    plot.savefig(plot_name)
    plot.close()

    # When Autocorrelation MA
    # Partial for AR


def total_process(evaluate_var='DUMMY_30_DAY', stat_y=False, stat_x=False, grenger=False, corr=False, stepwise=False,
                  stat_res=False,
                  serial_corr=False, ar_ma=False, insignificant=False, vif=False):
    df = pd.read_csv(STRING.final_file, sep=';', dtype={'FECHA': str, 'ANIO': str, 'MES': str, 'DIA': str})
    df = process_df(df)
    del df['PPORTUGAL']

    # PROCESS DF

    if stat_y:
        # 1) ESTACIONARIEDAD DEPENDIENTE-----------------------------------------------------------------------------
        # PRECIO ESPANIA
        test_stationarity(df['PESPANIA'], plot_show=True, plot_name='stationarity_pespania_dia.png')

        # DIFERENCIA PESPANIA
        df['D_PESPANIA'] = df['PESPANIA'] - df['PESPANIA'].shift(1)
        df = df.dropna(subset=['D_PESPANIA'])
        test_stationarity(df['D_PESPANIA'], plot_show=True, plot_name='stationarity_first_difference.png')


    if stat_x:
        # 2) ESTACIONARIEDAD DEMAS VARIABLES------------------------------------------------------------------------
        columns = df.columns.values.tolist()
        evaluation = ['DUMMY', 'DUMMY_5_DAY', 'DUMMY_10_DAY', 'DUMMY_20_DAY', 'DUMMY_15_DAY', 'DUMMY_30_DAY']
        evaluation.remove(evaluate_var)

        remove_list = ['PESPANIA', 'ANIO', 'FECHA', 'MES', 'DIA'] + evaluation
        columns = [x for x in columns if x not in remove_list]
        for i in remove_list:
            del df[i]

        for i in columns:
            print(i)
            test_stationarity(df[i], plot_show=False)

        need_differenciation = ['TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR', 'CICLO_COMBINADO', 'FUEL_PRIMA',
                                'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME_MADRID', 'TMAX_MADRID', 'TME_BCN',
                                'TMAX_BCN', 'TMIN_BCN', 'GDP']

        for i in need_differenciation:
            print(i)
            df_diff = df[i] - df[i].shift(1)
            df_diff = df_diff.dropna()
            test_stationarity(df_diff, plot_show=False)

        for i in need_differenciation:
            name = 'D_' + str(i)
            df[name] = df[i] - df[i].shift(1)
            del df[i]

        df = df.dropna()

    if grenger:
        # 3) CAUSALIDAD DE GRANGER

        columns = df.columns.values.tolist()
        columns.remove('D_PESPANIA')
        for i in columns:
            print(i)
            granger_causality(df, i, 'D_PESPANIA')

        reverse_causality = ['TOTAL_DEMANDA_NAC_ES']

        for i in reverse_causality:
            del df[i]

    if corr:
        # 4) CORRELATION VARIABLES
        correlation_get_all(df, get_all=True, show_plot=True)
        correlation_get_all(df, get_all=False)

    if stepwise:
        # 5) STEPWISE

        y = df['D_PESPANIA'].values.tolist()
        x = df.copy()
        del x['D_PESPANIA']
        del x[evaluate_var]
        names = x.columns.values.tolist()
        x = x.values.tolist()

        best_attributes = stepwise_reg.stepwise_regression.setpwise_reg(x, y, names)
        '''
        best_attirbutes = ['D_TOTAL_PRODUCCION_POR', 'REG_ESPECIAL', 'D_TOTAL_DEMANDA_POR', 'HIDRAULICA_CONVENC',
                           'D_FUEL_PRIMA', 'CARBON NACIONAL', 'CARBON_IMPO',
                           'TOTAL_DEMANDA_NAC_ES', '%EOLICA', 'NUCLEAR',
                           'TOTAL_PRODUCCION_ES', evaluate_var, 'D_TMAX_MADRID', 'D_TME_MADRID',
                           'WORKDAY', 'D_PESPANIA'
                           ]
        '''
        best_attributes += ['D_PESPANIA', evaluate_var]
        df = df[best_attributes]

        mod = sm.OLS(df['D_PESPANIA'], df.drop('D_PESPANIA', axis=1))
        res = mod.fit()
        result = res.summary()
        print(result)

    if stat_res:
        # 6) ESTACIONARIEDAD DE LOS RESIDUOS
        error = get_residuals(df.drop('D_PESPANIA', axis=1), df['D_PESPANIA'])
        test_stationarity(error['error'], plot_show=True, plot_name='stationarity_error.png')

    if serial_corr:
        # 7) EXISTS SERIAL CORRELATION EN LOS RESIDUOS?
        serial_correlation(error)

    if ar_ma:
        # 8) Choosing AR-MA (PACF-ACF)
        selecting_AR_MA(df['D_PESPANIA'])
        '''If you want evaluate individually
        lags_ar = 0
        lags_ma = 0
        lags_dda = 1
        dda_var = ['TOTAL_DEMANDA_NAC_ES', 'D_TOTAL_PRODUCCION_POR', 'TOTAL_PRODUCCION_ES']
        df = df.reset_index(drop=True)
        for i in range(1, lags_ma + 1, 1):
            name_error = 'MA_' + str(i)
            error[name_error] = error['error'].shift(i)
            df = pd.concat([df, error[name_error]], axis=1)
        for i in range(1, lags_ar + 1, 1):
            name_ar = 'AR_' + str(i)
            df[name_ar] = df['D_PESPANIA'].shift(i)
        for j in dda_var:
            if j in df.columns.values.tolist():
                for i in range(1, lags_dda + 1, 1):
                    name = str(j) + '_' + str(i)
                    df[name] = df[j].shift(i)

        df = df.dropna()

        error_model = get_residuals(df.drop('D_PESPANIA', axis=1).values, df['D_PESPANIA'].values)
        serial_correlation(error_model)

        mod = sm.OLS(df['D_PESPANIA'], df.drop('D_PESPANIA', axis=1))
        res = mod.fit()
        result = res.summary()
        print(result)
        '''

        df.index = df.index.to_datetime()
        model = sm.tsa.statespace.SARIMAX(df['D_PESPANIA'], exog=df.drop('D_PESPANIA', axis=1),
                                          order=(28, 0, 7), trend=None, enforce_invertibility=False,
                                          enforce_stationarity=False)
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        # plot.show()
        residuals.plot(kind='kde')
        # plot.show()
        print(residuals.describe())

    if insignificant:
        ins_vars = ['TOTAL_POT_IND_ES', 'TMIN_MADRID', 'CICLO_COMBINADO_1', 'TMIN_BCN_1', 'NUCLEAR',
                    'WORKDAY', 'FUEL_SIN_PRIMA', 'PRICE_OIL_1', 'GDP_1']
        for i in ins_vars:
            del df[i]

        df.index = df.index.to_datetime()
        model = sm.tsa.statespace.SARIMAX(df['D_PESPANIA'], exog=df.drop('D_PESPANIA', axis=1),
                                          order=(28, 0, 7), trend=None, enforce_invertibility=False,
                                          enforce_stationarity=False)
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        # plot.show()
        residuals.plot(kind='kde')
        # plot.show()
        print(residuals.describe())

        test_stationarity(residuals[0], plot_show=False)
        serial_correlation(residuals[0])

    if vif:
        # 9) MULTICOLINEALIDAD: ELIMINAR VIF > 10
        del_var = ['REG_ESPECIAL', '%EOLICA', 'TOTAL_DEMANDA_POR_1']
        for i in del_var:
            del df[i]

        vif = pd.DataFrame()
        vif['vif'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        vif['features'] = df.columns
        print(vif)

        df.index = df.index.to_datetime()
        model = sm.tsa.statespace.SARIMAX(df['D_PESPANIA'], exog=df.drop('D_PESPANIA', axis=1),
                                          order=(28, 0, 7), trend=None, enforce_invertibility=False,
                                          enforce_stationarity=False)
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        # plot.show()
        residuals.plot(kind='kde')
        # plot.show()
        print(residuals.describe())
        test_stationarity(residuals[0], plot_show=False)
        serial_correlation(residuals[0])

if __name__ == '__main__':

    total_process(evaluate_var='DUMMY_30_DAY', stat_y=True, stat_x=False, grenger=False, corr=False, stepwise=False,
                  stat_res=False, serial_corr=False, ar_ma=False, insignificant=False, vif=False)

    df = pd.read_csv(STRING.final_file, sep=';', dtype={'FECHA': str, 'ANIO': str, 'MES': str, 'DIA': str})
    df = process_df(df)
    plot_basics(df)



